# photo-cull

A command-line tool that uses local AI to help photographers pick their best
shots. It combines blur detection, perceptual hashing, and a locally running
vision model to reduce a folder of RAW or JPEG images to a curated set of
keepers. Everything runs locally, no data leaving your machine.

## How it works

* Extract EXIF timestamps using `exiftool` and sort images chronologically
* Load each image. RAW files are decoded in memory using `rawpy`, no
intermediate files written to disk
* Flag blurry images using Laplacian variance (`opencv-python`)
* Group consecutive shots into burst clusters based on timestamp proximity
(within 2 seconds)
* Within each cluster, identify near-duplicates using perceptual hashing
* Send each duplicate group to a locally running vision model. The model
picks the sharpest, best-composed keeper and gives a reason
* Copy keepers to an output folder and write a progressive CSV report with
every decision

## Requirements

### System

* [ExifTool](https://exiftool.org)
* [llama-server](https://llama-cpp.com)
* [Gemma 4 12B GGUF model and its mmproj file](https://huggingface.co/ggml-org/gemma-4-12B-it-GGUF/tree/main)

### Python

```
pip install opencv-python numpy imagehash rawpy Pillow requests
```

## Usage

Start llama-server with the Gemma 4 12B model and its mmproj.

```
% llama-server \
  -m /path/to/gemma-4-12b-it-Q4_K_M.gguf \
  --mmproj /path/to/mmproj-gemma-4-12B-it-Q8_0.gguf \
  --no-jinja --chat-template gemma \
  --image-min-tokens 280 --image-max-tokens 280 \
  --ubatch-size 2048 --batch-size 2048
```

Then run the script.

```
% python photo_cull.py --input /path/to/photos
```

With all options:

```
% python photo_cull.py \
  --input /path/to/photos \
  --output /path/to/keepers \
  --report /path/to/cull_report.csv \
  --blur-threshold 100 \
  --burst-gap 2.0 \
  --model gemma-4-12b-it-q4_k_m.gguf \
  --llama-server http://localhost:8080
```

While the script is running you can watch progress in a second terminal.

```
% tail -f /path/to/cull_report.csv
```

## Output

Keepers are copied to the output directory (default: `keepers/` under the input
directory). A CSV report is written progressively with one row per image.

| **Field**          | **Description**                                        |
|:-------------------|:-------------------------------------------------------|
| filename           | Original filename                                      |
| cluster_id         | Burst cluster the image belongs to                     |
| blur_score         | Laplacian variance score - lower means blurrier        |
| is_blurry          | True if below blur threshold                           |
| is_duplicate       | True if near-duplicate of another image in the cluster |
| keeper             | True if selected as the keeper                         |
| reason             | Reason for keeping or discarding                       |

## Tested on

* Apple M2 Max, 64GB unified memory, macOS
* 166 RAW files (Canon CR2) processed in ~6 minutes

## RAW formats supported

* Canon `.cr2`
* Nikon `.nef`
* Fuji `.raf`
* Sony `.arw`
* Adobe `.dng`
* Olympus `.orf`
* Panasonic `.rw2`

## Ideas for contribution

* `--location` flag to pass shooting location to the model prompt, improving
subject identification
* Support for JPEG-only workflows (without rawpy)
* Configurable model prompt for different photography contexts (birds,
portraits, landscapes)
