from contextlib import redirect_stdout
from io import BytesIO
from PIL import Image
import piexif
import piexif.helper
import argparse
import time
import json
import ast
import re
import hydrus_api
import prompt_parser
import stealth_pnginfo

# Options
option_print_tags_to_console = False
option_redirect_print_to_file = False
option_stealth_pnginfo_enable = True
option_stealth_pnginfo_alpha_only = True
option_convert_addnet = True # Converts 'AddNet Module: Type' and 'AddNet Model: Name(Hash)' into 'Type: Name' and 'Type Hash: Hash'
option_keep_addnet_after_conversion = False # Keeps 'AddNet Module: Type' and 'AddNet Model: Name(Hash)'
option_skip_addnet_weights = True # Don't add 'AddNet Weight' tags
option_convert_size = True # Converts 'Size: WxH' into 'Width: W' and 'Height: H' and 'Hires resize: WxH' into 'Hires width: W' and 'Hires height: H'
option_keep_size_after_conversion = True # Keeps 'Size: WxH' and 'Hires resize: WxH'
option_others_like_webui = True # Renames some namespaces to be like WebUI
option_others_convert_dimensions = True # Converts 'Width: W' and 'Height: H' into 'Size: WxH'
option_others_keep_dimensions_after_conversion = True # Keeps 'Width: W' and 'Height: H'
option_nai_keep_generation_time = False

parser = argparse.ArgumentParser(description="Hydrus Process AI Metadata - Extract AI metadata from images saved in Hydrus and send them back as notes and tags using the API.")
parser.add_argument('-k', '--api-key', type=str, required=True, help="Hydrus API key (Required)")
parser.add_argument('-a', '--api-url', type=str, default=hydrus_api.DEFAULT_API_URL, help="Hydrus API URL (Default: 'http://127.0.0.1:45869)'")
parser.add_argument('-s', '--service', type=str, default='my tags', help="Tag service name (Default: 'my tags')")
parser.add_argument('-hf', '--hash-file', type=str, default='hashes.txt', help="Path to text file containing SHA-256 hashes (can be prefixed) (Default: 'hashes.txt')")
args = parser.parse_args()

# Initialize Hydrus client and get tag service key
hc = hydrus_api.Client(args.api_key, args.api_url)
tag_service_key = hc.get_service(service_name=args.service).get("service", {}).get("service_key", None)

Image.MAX_IMAGE_PIXELS = None # Turn off DecompressionBombWarning

def is_valid_hex_hash(hex_str):
    try:
        bytes.fromhex(hex_str)
        return len(hex_str) == 64 # Ensure it's a SHA-256 hash
    except ValueError:
        return False

def get_image_from_hydrus(file_hash):
    try:
        response = hc.get_file(file_hash)
        file_bytes = response.content # Access the content of the response to get the raw bytes
        return Image.open(BytesIO(file_bytes))
    except Exception as e:
        print(f"Error fetching image {file_hash} from Hydrus: {e}")
    return None

def get_image_info(image):
    try:
        info_data = image.info
        return info_data
    except Exception as e:
        print(f"Error extracting image info: {e}")
    return None

def get_metadata_from_image_info(image_info):
    if isinstance(image_info, str):
        image_info = load_json(image_info)

        if isinstance(image_info, str):
            if "\nSteps: " in image_info or image_info.startswith("Steps: "):
                return "A1111 WebUI", image_info
            else:
                return "", None

    if isinstance(image_info, dict):
        if "parameters" in image_info:
            parameters = image_info.get("parameters", None)
            if parameters:
                if '"sui_image_params":' in parameters:
                    return "SwarmUI", parameters
                elif "\nSteps: " in parameters or parameters.startswith("Steps: "):
                    return "A1111 WebUI", parameters
        if "prompt" and "workflow" in image_info:
            return "ComfyUI", image_info.get("prompt", None)
        if "prompt" and "comfyBoxWorkflow" in image_info:
            return "ComfyBox", image_info.get("prompt", None)
        if "prompt" in image_info:
            return "ComfyUI (?)", image_info.get("prompt", None)
        if "Comment" in image_info and image_info.get("Software", None) == "NovelAI":
            return "NovelAI", {key: image_info[key] for key in ('Title','Description','Software','Source','Generation time','Generation_time','Comment') if key in image_info}
        if "comment" in image_info:
            comment_data = image_info.get("comment", None)
            if comment_data:
                if isinstance(comment_data, bytes):
                    comment_data = comment_data.decode('utf-8', 'ignore')
                metadata_type, metadata = get_metadata_from_image_info(comment_data)
                if metadata_type: return metadata_type, metadata
        if "exif" in image_info:
            try:
                exif_data = piexif.load(image_info.get("exif", None))
                exif_data = (exif_data or {}).get("Exif", {}).get(piexif.ExifIFD.UserComment, b'')
                if exif_data:
                    try:
                        exif_data = piexif.helper.UserComment.load(exif_data)
                    except ValueError:
                        exif_data = exif_data.decode('utf-8', 'ignore')
                    metadata_type, metadata = get_metadata_from_image_info(exif_data)
                    if metadata_type: return metadata_type, metadata
            except Exception as e:
                print(f"Error processing exif data: {e}")

    return "", None

def save_tags_to_hydrus(file_hash, tags, tag_service_key):
    try:
        service_tags = {tag_service_key: tags}
        hc.add_tags(hashes=[file_hash], service_keys_to_tags=service_tags)
    except Exception as e:
        print(f"Error saving tags to Hydrus for {file_hash}: {e}")

def save_note_to_hydrus(file_hash, note_name, note_content):
    try:
        hc.set_notes(notes={note_name: note_content}, hash_=file_hash)
    except Exception as e:
        print(f"Error saving note to Hydrus for {file_hash}: {e}")

def format_prompt(prompt, steps):
    # Extract extra networks and remove them from the prompt
    re_extra_networks = re.compile(r'<([^:<>]+:[^:<>]+)(?::[^<>]+)?>')
    extra_networks = []
    modified_prompt = []
    for line in prompt:
        extra_networks.extend(re.findall(re_extra_networks, line))
        modified_prompt.append(re.sub(re_extra_networks, '', line))
    prompt = modified_prompt

    tags = []
    modified_prompt = []
    for line in prompt_parser.get_learned_conditioning_prompt_schedules(prompt, steps):
        if len(line) > 1:
            tags.append('prompt editing:true')
        for step, prompt_part in line:
            prompt_part = re.sub(r'(?<!\\)[{}]', '', prompt_part) # Remove '{' and '}'

            tokens = []
            for token, weight in prompt_parser.parse_prompt_attention(prompt_part):
                if token == "BREAK":
                    token = " BREAK "
                tokens.append(token)
            modified_prompt.append(''.join(tokens))
    prompt = modified_prompt

    tags.extend(split_into_tags(prompt))
    tags.extend(extra_networks)

    return tags

def split_into_tags(prompt):
    tags = []
    for line in prompt:
        tags.extend(re.split(r',|(?<!\d)\.(?!\d)|\bAND\b|\bBREAK\b|\bADDBASE\b|\bADDCOMM\b|\bADDCOL\b|\bADDROW\b', line))
    return [tag.strip() for tag in tags if tag.strip()]

def format_settings(settings_params, steps):
    pairs = []
    addnet_module = None

    for key, value in parse_key_value_pairs('\n'.join(settings_params).lower()):
        if 'prompt' in key or 'negative' in key:
            if re.search(r' (negative )?prompt( \d+(st|nd|rd|th))?$|^sv_(prompt|negative)$', key):
                continue
        if value.startswith('"') and value.endswith('"'):
            if key in ('x values','y values','z values'):
                continue
            value = value[1:-1]
            if ':' in value:
                if value.startswith('{') and value.endswith('}'):
                    value = value.replace('true', 'True').replace('false', 'False').replace('none', 'None')
                    json_value = load_dict_from_string(value, key)
                    if json_value: pairs.extend(flatten_nested_dict(json_value, key, steps))
                else:
                    for k, v in parse_key_value_pairs(value):
                        if key == 'lora hashes':
                            pairs.append(f"lora:{k}")
                            pairs.append(f"lora hash:{v}")
                        elif key == 'ti hashes':
                            pairs.append(f"ti:{k}")
                            pairs.append(f"ti hash:{v}")
                        else:
                            pairs.append(f"{key} {k}:{v}")
            else:
                pairs.append(f"{key}:{value}")
        elif value.startswith('{') and value.endswith('}'):
            json_value = load_dict_from_string(value, key)
            if json_value: pairs.extend(flatten_nested_dict(json_value, key, steps))
        elif option_convert_addnet and 'addnet module' in key:
            if option_keep_addnet_after_conversion: pairs.append(f"{key}:{value}")
            addnet_module = value
        elif option_convert_addnet and 'addnet model' in key:
            if addnet_module:
                if option_keep_addnet_after_conversion: pairs.append(f"{key}:{value}")
                addnet_model, addnet_hash = re.search(r'^(.+?)\((\w+)\)$', value).groups()
                pairs.append(f"{addnet_module}:{addnet_model}")
                pairs.append(f"{addnet_module} hash:{addnet_hash}")
                addnet_module = None
            else:
                pairs.append(f"{key}:{value}")
        elif 'addnet weight' in key:
            if not option_skip_addnet_weights: pairs.append(f"{key}:{value}")
            if 'addnet weight a' in key:
                pairs.append("addnet separate weights:true")
        elif option_convert_size and key in ('size','hires resize'):
            if option_keep_size_after_conversion: pairs.append(f"{key}:{value}")
            sizew, sizeh = re.search(r'^(\d+)x(\d+)$', value).groups()
            pairs.append(f"width:{sizew}" if key == 'size' else f"hires width:{sizew}")
            pairs.append(f"height:{sizeh}" if key == 'size' else f"hires height:{sizeh}")
        else:
            pairs.append(f"{key}:{value}")

    return [tag.strip() for tag in pairs if tag.strip()]

def load_json(input_string):
    try:
        result = json.loads(input_string)
        return result
    except json.JSONDecodeError:
        return input_string

def load_dict_from_string(input_string, key="A"):
    input_string = load_json(input_string)

    if isinstance(input_string, dict):
        return input_string
    elif isinstance(input_string, str):
        try:
            input_string = ast.literal_eval(input_string)

            if isinstance(input_string, dict):
                return input_string
            else:
                print(f"Error: {key} value was detected as python dictionary, but it is not a dictionary: {input_string}")
        except (ValueError, SyntaxError):
            print(f"Error: {key} value was detected as json or python dictionary, but it couldn't be decoded: {input_string}")
    else:
        print(f"Error: {key} value was detected as json, but it is not a dictionary: {input_string}")
    return None

def parse_key_value_pairs(input_string):
    pairs = []
    key = None
    value = ""
    brace_count = 0
    in_string = False
    escaping = False

    for i, char in enumerate(input_string):
        if char == '"' and not escaping:
            in_string = not in_string
        elif char == '\\' and not escaping:
            escaping = True
        else:
            escaping = False

        if char == '{' and not in_string:
            brace_count += 1
        elif char == '}' and not in_string:
            brace_count -= 1

        if (char == ',' and brace_count == 0 and not in_string) or i == len(input_string) - 1:
            if i == len(input_string) - 1:
                value += char
            if key is not None and value:
                pairs.append((key, value))
            elif key is None and value and pairs:
                prev_key, prev_value = pairs.pop()
                pairs.append((prev_key, f"{prev_value},{value}"))
            key, value = None, ""
        elif char == ':' and brace_count == 0 and not in_string:
            key, value = value, ""
        else:
            value += char

    return [(k.strip(), v.strip()) for k, v in pairs]

def flatten_nested_dict(dictionary, prefix="", steps=20):
    if prefix: prefix += " "

    result = []
    for k, v in dictionary.items():
        if isinstance(v, dict):
            result.extend(flatten_nested_dict(v, prefix + k, steps))
        elif k == 'prompt':
            result.extend(format_prompt(v.replace('\\"', '"').split('\n'), steps))
            result.append(f"{prefix}{k}:true")
        elif k == 'neg_prompt':
            result.extend([f"negative:{tag}" for tag in format_prompt(v.replace('\\"', '"').split('\n'), steps)])
            result.append(f"{prefix}{k}:true")
        else:
            result.append(f"{prefix}{k}:{v}")

    return result

def format_webui(metadata):
    lines = metadata.split('\n')
    positive_prompt = []
    negative_prompt = []
    settings_params = []
    current_section = 'positive'

    for line in lines:
        if current_section == 'positive' and line.startswith("Negative prompt: "):
            current_section = 'negative'
            line = line[len("Negative prompt: "):]
        elif current_section != 'settings' and line.startswith("Steps: "):
            current_section = 'settings'
            steps = re.search(r'(?<=^Steps: )\d+(?=,)', line)
            steps = int(steps.group()) if steps else 20
        elif current_section == 'settings' and (line.startswith("Template: ") or line.startswith("Negative Template: ")):
            break

        if current_section == 'positive':
            positive_prompt.append(line)
        elif current_section == 'negative':
            negative_prompt.append(line)
        elif current_section == 'settings':
            settings_params.append(line)

    if not settings_params:
        print("Error: Line with steps not found, skipping to the next file.")
        return

    settings_params = format_settings(settings_params, steps)
    positive_prompt = format_prompt(positive_prompt, steps)
    negative_prompt = format_prompt(negative_prompt, steps)

    negative_prompt = [f"negative:{tag}" for tag in negative_prompt]

    tags = positive_prompt + negative_prompt + settings_params
    tags.append('prompt type:a1111')

    if option_print_tags_to_console: print_tags(tags)

    return tags

def format_comfyui(metadata, comfybox=False):
    metadata = load_json(metadata)

    if not isinstance(metadata, dict):
        print("Error: Failed to load json, skipping to the next file.")
        return

    settings_params = []
    positive_nodes = []
    negative_nodes = []
    positive_prompt = []
    negative_prompt = []
    width = height = None

    translation_dict = {
        'cfg': 'cfg scale',
        'denoise': 'denoising strength',
        'sampler_name': 'sampler',
        'ckpt_name': 'model',
        'vae_name': 'vae',
        'lora_name': 'lora'
    }

    for id, node in metadata.items():
        if node['class_type'] in ('KSampler','KSamplerAdvanced','EmptyLatentImage','LatentUpscale','LatentUpscaleBy'):
            empty_image = True if node['class_type'] == 'EmptyLatentImage' else False
            for key, value in node['inputs'].items():
                if not isinstance(value, list):
                    if key == 'denoise' and value == 1:
                        continue
                    if key in ('width','height'):
                        if option_others_convert_dimensions and key == 'width': width = value
                        if option_others_convert_dimensions and key == 'height': height = value
                        if option_others_convert_dimensions and width and height:
                            settings_params.append(f"size:{width}x{height}" if empty_image else f"upscale size:{width}x{height}")
                            width = height = None
                        if option_others_convert_dimensions and not option_others_keep_dimensions_after_conversion:
                            continue
                        if not empty_image: key = 'upscale ' + key
                    new_key = translation_dict.get(key, key) if option_others_like_webui else key
                    settings_params.append(f"{new_key}:{value}".replace('_', ' '))
                else:
                    if key == 'positive':
                        positive_nodes.append(value[0])
                    elif key == 'negative':
                        negative_nodes.append(value[0])
        elif node['class_type'] in ('CheckpointLoaderSimple','VAELoader','LoraLoader'):
            for key, value in node['inputs'].items():
                if key in ('ckpt_name','vae_name','lora_name'):
                    if '\\' in value or '/' in value:
                        value = re.sub(r'^.*[\\/]', '', value)
                    new_key = translation_dict.get(key, key) if option_others_like_webui else key
                    settings_params.append(f"{new_key}:{value}") # Remove extension to be like WebUI?

    for node in positive_nodes:
        for key in ('text','text_g','text_l','Text','input_id'):
            if key == 'input_id' and not metadata[node].get('class_type', None) == 'ComfyUIDeployExternalText':
                continue
            value = metadata[node]['inputs'].get(key, None)
            if isinstance(value, list):
                positive_nodes.append(value[0])
            elif isinstance(value, str):
                positive_prompt.append(value)

    for node in negative_nodes:
        for key in ('text','text_g','text_l','Text','input_id'):
            if key == 'input_id' and not metadata[node].get('class_type', None) == 'ComfyUIDeployExternalText':
                continue
            value = metadata[node]['inputs'].get(key, None)
            if isinstance(value, list):
                negative_nodes.append(value[0])
            elif isinstance(value, str):
                negative_prompt.append(value)

    positive_prompt = format_prompt(list(dict.fromkeys(positive_prompt)), 20)
    negative_prompt = format_prompt(list(dict.fromkeys(negative_prompt)), 20)

    negative_prompt = [f"negative:{tag}" for tag in negative_prompt]

    tags = settings_params + positive_prompt + negative_prompt
    tags.append('prompt type:comfybox' if comfybox else 'prompt type:comfyui')

    if option_print_tags_to_console: print_tags(tags)

    return tags

def format_novelai(metadata):
    tags = []

    if 'Source' in metadata: tags.append(f"Source:{metadata['Source']}")

    if option_nai_keep_generation_time:
        if 'Generation time' in metadata: tags.append(f"Generation time:{metadata['Generation time']}")
        if 'Generation_time' in metadata: tags.append(f"Generation time:{metadata['Generation_time']}")

    metadata = load_json(metadata['Comment'])

    if not isinstance(metadata, dict):
        print("Error: Failed to load json, skipping to the next file.")
        return

    positive_prompt = []
    negative_prompt = []
    settings_params = []

    translation_dict = {
        'scale': 'cfg scale'
    }

    metadata.pop('signed_hash', None) # Ignore this parameter

    positive_prompt = convert_novelai_to_webui(metadata.pop('prompt', '')).split('\n')
    negative_prompt = convert_novelai_to_webui(metadata.pop('uc', '')).split('\n')

    steps = metadata.get('steps', 20)

    if option_others_convert_dimensions and not option_others_keep_dimensions_after_conversion:
        width, height = metadata.pop('width', None), metadata.pop('height', None)
    else:
        width, height = metadata.get('width', None), metadata.get('height', None)

    if option_others_convert_dimensions and width and height:
        settings_params.append(f"size:{width}x{height}")

    for key, value in metadata.items():
        if not isinstance(value, list):
            new_key = translation_dict.get(key, key) if option_others_like_webui else key
            settings_params.append(f"{new_key}:{value}".replace('_', ' '))
        else:
            for list_val in value:
                new_key = translation_dict.get(key, key) if option_others_like_webui else key
                settings_params.append(f"{new_key}:{list_val}".replace('_', ' '))

    positive_prompt = format_prompt(positive_prompt, steps)
    negative_prompt = format_prompt(negative_prompt, steps)

    negative_prompt = [f"negative:{tag}" for tag in negative_prompt]

    tags.extend(positive_prompt + negative_prompt + settings_params)
    tags.append('prompt type:novelai')

    if option_print_tags_to_console: print_tags(tags)

    return tags

def format_swarmui(metadata):
    metadata = load_json(metadata)

    if not isinstance(metadata, dict):
        print("Error: Failed to load json, skipping to the next file.")
        return

    metadata = metadata.get('sui_image_params', None)

    positive_prompt = []
    negative_prompt = []
    settings_params = []

    translation_dict = {
        'cfgscale': 'cfg scale',
        'loras': 'lora'
    }

    for key in ('loraweights','date','generation_time'): metadata.pop(key, None) # Ignore these parameters

    positive_prompt = metadata.pop('prompt', '')
    positive_prompt = metadata.pop('original_prompt', positive_prompt)
    positive_prompt = positive_prompt.split('\n')
    negative_prompt = metadata.pop('negativeprompt', '').split('\n')

    steps = metadata.get('steps', 20)

    if option_others_convert_dimensions and not option_others_keep_dimensions_after_conversion:
        width, height = metadata.pop('width', None), metadata.pop('height', None)
    else:
        width, height = metadata.get('width', None), metadata.get('height', None)

    if option_others_convert_dimensions and width and height:
        settings_params.append(f"size:{width}x{height}")

    for key, value in metadata.items():
        if value:
            if not isinstance(value, list):
                if key in ('model','vae'):
                    if '\\' in value or '/' in value:
                        value = re.sub(r'^.*[\\/]', '', value)
                new_key = translation_dict.get(key, key) if option_others_like_webui else key
                settings_params.append(f"{new_key}:{value}".replace('_', ' '))
            else:
                for list_val in value:
                    if key == 'loras':
                        if '\\' in list_val or '/' in list_val:
                            list_val = re.sub(r'^.*[\\/]', '', list_val)
                    new_key = translation_dict.get(key, key) if option_others_like_webui else key
                    settings_params.append(f"{new_key}:{list_val}".replace('_', ' '))

    positive_prompt = format_prompt(positive_prompt, steps)
    negative_prompt = format_prompt(negative_prompt, steps)

    negative_prompt = [f"negative:{tag}" for tag in negative_prompt]

    tags = positive_prompt + negative_prompt + settings_params
    tags.append('prompt type:swarmui')

    if option_print_tags_to_console: print_tags(tags)

    return tags

def convert_novelai_to_webui(prompt):
    prompt = prompt.replace("(", "\\(")
    prompt = prompt.replace(")", "\\)")
    prompt = prompt.replace("{", "(")
    prompt = prompt.replace("}", ")")

    return prompt

def print_tags(tags):
    tags = [' '.join(tag.split()) for tag in tags] # Remove repeated whitespace
    tags = list(dict.fromkeys(tags)) # Remove duplicates
    print("\nTags:")
    print("\n".join(tags), "\n")

def process_images_from_hashes(file_path):
    with open(file_path, 'r') as f:
        hashes = f.read().splitlines()

    print("Preparing hashes...")

    processed_hashes = []
    for file_hash in hashes:
        file_hash = file_hash.removeprefix('sha256:')

        if is_valid_hex_hash(file_hash):
            processed_hashes.append(file_hash)
        else:
            print(f"Invalid hash format: {file_hash}")
    hashes = list(dict.fromkeys(processed_hashes)) # Remove duplicates

    print("\nProcessing...")

    hash_count = len(hashes)
    hash_count_digits = len(str(hash_count))

    for i, file_hash in enumerate(hashes):
        image = get_image_from_hydrus(file_hash)

        if image:
            metadata_type, metadata = get_metadata_from_image_info(get_image_info(image))

            stealth_flag = ""
            if option_stealth_pnginfo_enable and not metadata and image.format.lower() == 'png':
                if option_stealth_pnginfo_alpha_only:
                    if image.mode == 'RGBA' and image.getchannel('A').getextrema()[0] < 255:
                        metadata = stealth_pnginfo.read_info_from_image_stealth(image)
                elif image.mode == 'RGB' or image.mode == 'RGBA':
                    metadata = stealth_pnginfo.read_info_from_image_stealth(image)

                if metadata:
                    metadata_type, metadata = get_metadata_from_image_info(metadata)
                    stealth_flag = " (Stealth)"

            if metadata:
                if metadata_type == "A1111 WebUI":
                    print(f" - {i+1:0{hash_count_digits}d}/{hash_count} - {file_hash}: Type: {metadata_type}{stealth_flag}")
                    tags = format_webui(metadata)
                elif metadata_type == "ComfyUI" or metadata_type == "ComfyUI (?)":
                    print(f" - {i+1:0{hash_count_digits}d}/{hash_count} - {file_hash}: Type: {metadata_type}{stealth_flag}")
                    tags = format_comfyui(metadata)
                elif metadata_type == "ComfyBox":
                    print(f" - {i+1:0{hash_count_digits}d}/{hash_count} - {file_hash}: Type: {metadata_type}{stealth_flag}")
                    tags = format_comfyui(metadata, True)
                elif metadata_type == "NovelAI":
                    print(f" - {i+1:0{hash_count_digits}d}/{hash_count} - {file_hash}: Type: {metadata_type}{stealth_flag}")
                    tags = format_novelai(metadata)
                elif metadata_type == "SwarmUI":
                    print(f" - {i+1:0{hash_count_digits}d}/{hash_count} - {file_hash}: Type: {metadata_type}{stealth_flag}")
                    tags = format_swarmui(metadata)

                save_tags_to_hydrus(file_hash, tags, tag_service_key)
                save_note_to_hydrus(file_hash, "parameters", metadata if isinstance(metadata, str) else json.dumps(metadata))
            else:
                print(f" - {i+1:0{hash_count_digits}d}/{hash_count} - {file_hash}: No parameters found")
        else:
            print(f" - {i+1:0{hash_count_digits}d}/{hash_count} - {file_hash}: Image not found")

t1 = time.time()
if option_redirect_print_to_file:
    with open('debug.txt', 'w', encoding="utf-8") as f:
        with redirect_stdout(f):
            process_images_from_hashes(args.hash_file)
else:
    process_images_from_hashes(args.hash_file)
t2 = time.time()
print(f"\nFinished in: {t2-t1} seconds")
