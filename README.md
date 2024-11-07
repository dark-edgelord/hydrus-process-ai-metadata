# Hydrus Process AI Metadata

A Python script that extracts AI metadata from images saved in Hydrus Network and sends them back as notes and tags using the API.\
Inspired by another script (https://github.com/space-nuko/sd-webui-utilities/blob/master/import_to_hydrus.py), which is a bit outdated now. I also didn't like a few things so I decided to try making my own.

## Main Features

- Supports metadata formats generated from A1111 WebUI, ComfyUI, ComfyBox, SwarmUI and NovelAI.
- Stealth-pnginfo support (with an option to only check images with an alpha layer, as rgb mode can take too much time).
- Supports PNG, JPEG and maybe other image formats too.
- Options to customize results like renaming some namespaces of other UIs to WebUI namespaces.

## Installation

### Prerequisites

- Python 3.6 or higher

### Clone The Repository

```bash
git clone https://github.com/dark-edgelord/hydrus-process-ai-metadata
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

- Create a `hashes.txt` file in the script's folder.
- Select your images in Hydrus and copy their SHA256 hashes by right-clicking them and selecting `Share > Copy Hashes > SHA256`.
- Paste the hashes into the `hashes.txt` file and save (it's ok if the hashes have a prefix).
- Run the script using the following command-line arguments:

```bash
python hydrus_process_ai_metadata.py -k HYDRUS_API_KEY [-a HYDRUS_API_URL] [-s TAG_SERVICE] [-hf HASH_FILE]
```

### Command-Line Arguments

- `-k`, `--api-key` (Required): Hydrus API key.
- `-a`, `--api-url` (Optional): Hydrus API URL (Default: 'http://127.0.0.1:45869').
- `-s`, `--service` (Optional): Tag service name (Default: 'my tags').
- `-hf`, `--hash-file` (Optional): Path to text file containing SHA-256 hashes (Default: 'hashes.txt').

### Examples

Import tags to the `my tags` tag service through a default Hydrus API URL using hashes from `hashes.txt`:

```bash
python hydrus_process_ai_metadata.py -k 12345
```

Import tags to the `ai metadata` tag service through a custom Hydrus API URL using hashes from `my_hashes.txt` that's in another folder:

```bash
python hydrus_process_ai_metadata.py -k 12345 -a http://127.0.0.1:45870 -s "ai metadata" -hf path\my_hashes.txt
```

### Options

There are some options in the main script near the top, which can change the script's behavior in some way. I'll probably put them to a separate file in the future, but for now the main script has to be edited.

## New Features/Differences

- Works with files that are already in Hydrus using hashes as selection (pasted as a list to a txt file) instead of needing to import files from a folder.
- Support for metadata stored in EXIF UserComment (mainly for JPEG).
- Separate negative tags instead of the whole negative prompt as a single tag.
- Splits tags by regional prompter keywords (such as ADDCOL, ADDROW etc.).
- Better overall parsing of settings from addons and other nested settings like lora hashes (WebUI).
- Nested setting namespaces are prefixed with parent namespace.
- Sends tags for a file immediately instead of pooling 100 first (prevents progress loss in case of a potential crash).
- Fixes a bunch of crash cases like when metadata has no negative prompt or when image is indexed instead of rgb during stealth-pnginfo check etc.
- Only one main parameters note is saved (no file path note or separate positive/negative prompt notes).
- Lightweight, no crazy requirements needed.
- Ignores extra lines after settings for WebUI (usually templates and some rare garbage I didn't know how to parse).
- Adds tiled diffusion region prompt/negative tags as normal tags.
- Skips adding adetailer/ddetailer/hires/unprompted/wildcard prompt/negative tags, also sv_prompt and sv_negative (no idea what these two are). Maybe I could add them as normal tags in the future as an option though.
- And more parsing changes, which I stopped writing down as the script started to become too different.

## Potential Issues

- Very large files fail to be fetched through the Hydrus API.
- Will badly parse prompt editing syntax `[from:to:when]` if it contains an extra `:` (user error). Will only remove square brackets, but will keep everything inside as a tag.
- Tags with some special characters may also be parsed badly, but I tried to make sure it doesn't happen.
- NovelAI parser may badly parse the prompt if it contains something that resembles prompt editing or alternating words, as it uses the WebUI parser.
- SwarmUI parser won't parse its `<>` advanced syntax properly (not supported).
- It's possible some tags may turn out worse or just different than from the other script. It's hard to compare as the results are very different.

## Some Explanations

### Why hash matching existing files instead of importing?

I find it more comfortable to import my files into Hydrus first and then have them tagged. This has the benefit of being able to quickly tag files downloaded from the internet straight into Hydrus or retag something after a change in the script, which doesn't require exporting the files in a temporary folder, running the script and then deleting the files. It's also faster in this case as the script doesn't need to calculate hashes for each file every time you run it (which takes time especially for big files). It also gives me more control over the import itself.

### Why separate negative tags?

Preference. I think it's less ugly and it makes searching easier. The namespace can be hidden from display if it's too spammy anyway, which the alternative was too in thumbnail view, as each file had its own unique negative (mostly).
