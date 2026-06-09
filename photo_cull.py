"""
photo_cull.py - cull burst photos using blur detection, perceptual hashing,
and a local vision model (Gemma 4 12B via llama-server).

usage:
    python photo_cull.py --input /path/to/jpegs [--output /path/to/keepers]
                         [--report /path/to/jpegs/cull_report.csv]
                         [--blur-threshold 100] [--burst-gap 2.0]
                         [--model gemma-4-12b-it-q4_k_m.gguf]
                         [--llama-server http://localhost:8080]

    If the output directory isn't passed, the default is keepers under the
    input directory.

    The report is written to cull_report.csv under the output directory by
    default.
"""

import argparse
import base64
import csv
import shutil
import sys
from datetime import datetime
from io import BytesIO
from pathlib import Path
import subprocess

import cv2
import imagehash
import requests
from PIL import Image
import rawpy

# defaults
DEFAULT_BLUR_THRESHOLD = 100.0
DEFAULT_BURST_GAP = 2.0
DEFAULT_VISION_MODEL = "gemma-4-12b-it-q4_k_m.gguf"
DEFAULT_LLAMA_SERVER = "http://localhost:8080"
DEFAULT_REPORT_FILE = "cull_report.csv"
DEFAULT_OUTPUT_DIR = "keepers"
MAX_IMAGE_SIZE = 1024
PHASH_THRESHOLD = 10

JPEG_EXTENSIONS = {".jpg", ".jpeg"}
RAW_EXTENSIONS = {
    ".cr2",                     # Canon
    ".nef",                     # Nikon
    ".raf",                     # Fuji
    ".arw",                     # Sony
    ".dng",                     # Adobe
    ".orf",                     # Olympus
    ".rw2"                      # Panasonic
}
SUPPORTED_EXTENSIONS = JPEG_EXTENSIONS | RAW_EXTENSIONS


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Cull burst photos using blur detection, perceptual hashing, and a local vision model.")
    parser.add_argument("--input", required=True,
                        help="Input directory containing JPEG images")
    parser.add_argument("--output",
                        help="Output directory for keepers (default: keepers/ under input)")
    parser.add_argument("--report",
                        help="Path to CSV report (default: cull_report.csv under input)")
    parser.add_argument("--blur-threshold", type=float, default=DEFAULT_BLUR_THRESHOLD,
                        help=f"Laplacian variance threshold for blur detection (default: {DEFAULT_BLUR_THRESHOLD})")
    parser.add_argument("--burst-gap", type=float, default=DEFAULT_BURST_GAP,
                        help=f"Max seconds between shots to be considered a burst (default: {DEFAULT_BURST_GAP})")
    parser.add_argument("--model", default=DEFAULT_VISION_MODEL,
                        help=f"Model name for llama-server (default: {DEFAULT_VISION_MODEL})")
    parser.add_argument("--llama-server", default=DEFAULT_LLAMA_SERVER,
                        help=f"llama-server base URL (default: {DEFAULT_LLAMA_SERVER})")
    return parser.parse_args()


def get_timestamp(path: Path) -> datetime:
    """
    Extract DateTimeOriginal from EXIF using exiftool.

    Falls back to file modification time if not found.
    """
    try:
        result = subprocess.run(
            ["exiftool", "-DateTimeOriginal", "-s3", str(path)],
            capture_output=True, text=True
        )
        raw = result.stdout.strip()
        if raw:
            return datetime.strptime(raw, "%Y:%m:%d %H:%M:%S")
    except Exception:
        pass
    return datetime.fromtimestamp(path.stat().st_mtime)


def load_image(path: Path) -> Image.Image:
    """Load a JPEG or RAW file and return a resized PIL Image."""
    if path.suffix.lower() in RAW_EXTENSIONS:
        with rawpy.imread(str(path)) as raw:
            rgb = raw.postprocess(use_camera_wb=True)
        img = Image.fromarray(rgb)
    else:
        img = Image.open(path).convert("RGB")
    img.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE))
    return img


def blur_score(img: Image.Image) -> float:
    """Return the Laplacian variance of the image. Lower means blurrier."""
    try:
        gray = cv2.cvtColor(cv2.UMat(img), cv2.COLOR_RGB2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).get().var()
    except Exception:
        import numpy as np
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()


def encode_image(img: Image.Image) -> str:
    """Encode a PIL Image as a base64 JPEG data URI."""
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def load_images(input_dir: Path) -> list[tuple[Path, datetime]]:
    """Return a list of (path, timestamp) sorted chronologically."""
    images = []
    for path in input_dir.iterdir():
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            images.append((path, get_timestamp(path)))
    return sorted(images, key=lambda x: x[1])


def cluster_bursts(images: list[tuple[Path, datetime]], burst_gap: float) -> list[list[Path]]:
    """Group images into burst clusters based on timestamp proximity."""
    if not images:
        return []

    clusters = []
    current = [images[0][0]]

    for i in range(1, len(images)):
        gap = (images[i][1] - images[i - 1][1]).total_seconds()
        if gap <= burst_gap:
            current.append(images[i][0])
        else:
            clusters.append(current)
            current = [images[i][0]]
    clusters.append(current)
    return clusters


def find_duplicates(paths: list[Path], images: dict[Path, Image.Image]) -> list[list[Path]]:
    """
    Within a cluster, group near-duplicate images using perceptual hashing.

    Returns a list of duplicate groups, each group being a list of paths.
    """
    if len(paths) == 1:
        return [paths]

    hashes = {p: imagehash.phash(images[p]) for p in paths}
    visited = set()
    groups = []

    for i, p1 in enumerate(paths):
        if p1 in visited:
            continue
        group = [p1]
        visited.add(p1)
        for p2 in paths[i + 1:]:
            if p2 not in visited:
                if hashes[p1] - hashes[p2] <= PHASH_THRESHOLD:
                    group.append(p2)
                    visited.add(p2)
        groups.append(group)

    return groups


def clean_response(text: str) -> str:
    """Strip thinking tokens from Gemma 4 responses."""
    marker = "<channel|>"
    idx = text.find(marker)
    return text[idx + len(marker):].strip() if idx != -1 else text.strip()


def pick_keeper(paths: list[Path], images: dict[Path, Image.Image],
                model: str, server: str) -> tuple[str, str]:
    """
    Ask the vision model to pick the best keeper from a cluster.

    Returns (chosen_filename, reason).
    """
    filenames = [p.name for p in paths]

    # Build the prompt
    numbered = "\n".join(f"{i + 1}. {name}" for i, name in enumerate(filenames))
    question = (
        f"You are given {len(paths)} photos from a burst sequence:\n{numbered}\n\n"
        f"Examine each photo carefully and pick the single best keeper using these criteria in order:\n"
        f"1. The subject must be fully in frame — avoid photos where the animal is cut off at the edges\n"
        f"2. Sharpness of the subject's eye and facial features\n"
        f"3. Overall clarity and composition\n\n"
        f"If all photos have the subject cut off, pick the one with the sharpest eye.\n\n"
        f"Respond with exactly the filename and one sentence explaining why. "
        f"Example: IMG_0468.jpeg - subject fully in frame with sharpest eye detail."
    )

    content = []
    for i, path in enumerate(paths):
        content.append({
            "type": "text",
            "text": f"Image {i + 1}: {path.name}"
        })
        content.append({
            "type": "image_url",
            "image_url": {"url": encode_image(images[path])}
        })
    content.append({"type": "text", "text": question})

    try:
        r = requests.post(
            f"{server}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": content}]
            },
            timeout=120
        )
        response = r.json()
        if "choices" not in response:
            print(f"  model error: {response}", file=sys.stderr)
            return "model_error", str(response)

        text = clean_response(response["choices"][0]["message"]["content"])

        # validate: does the response contain one of our filenames?
        for filename in filenames:
            if filename in text:
                reason = text.replace(filename, "").strip().lstrip("-").strip()
                return filename, reason

        # model responded but didn't use a recognisable filename
        return "model_error", text

    except Exception as e:
        return "model_error", str(e)

def resolve_paths(args) -> tuple[Path, Path, Path]:
    """Resolve input, output, and report paths from args, applying defaults."""
    input_dir = Path(args.input)
    output_dir = Path(args.output) if args.output else input_dir / DEFAULT_OUTPUT_DIR
    report_path = Path(args.report) if args.report else input_dir / DEFAULT_OUTPUT_DIR / DEFAULT_REPORT_FILE
    return input_dir, output_dir, report_path


def cache_and_score(images_ts: list[tuple[Path, datetime]]
                    ) -> tuple[dict[Path, Image.Image], dict[Path, float]]:
    """Load every image into memory and compute its blur score."""
    image_cache: dict[Path, Image.Image] = {}
    blur_scores: dict[Path, float] = {}
    for path, _ in images_ts:
        img = load_image(path)
        image_cache[path] = img
        blur_scores[path] = blur_score(img)
    return image_cache, blur_scores


def write_row(writer, csvfile, *, filename, cluster_id, blur_score,
              is_blurry, is_duplicate, keeper, reason):
    """Write a single CSV row and flush."""
    writer.writerow({
        "filename": filename,
        "cluster_id": cluster_id,
        "blur_score": f"{blur_score:.1f}",
        "is_blurry": is_blurry,
        "is_duplicate": is_duplicate,
        "keeper": keeper,
        "reason": reason,
    })
    csvfile.flush()


def process_duplicate_group(group, cluster_id, blur_scores, image_cache,
                            output_dir, writer, csvfile, args):
    """Handle a single duplicate group: unique passes through, dupes go to model."""
    if len(group) == 1:
        path = group[0]
        print(f"  unique keeper: {path.name}")
        shutil.copy2(path, output_dir / path.name)
        write_row(writer, csvfile, filename=path.name, cluster_id=cluster_id,
                  blur_score=blur_scores[path], is_blurry=False, is_duplicate=False,
                  keeper=True, reason="unique in cluster")
        return

    print(f"  duplicate group of {len(group)}, asking model...")
    chosen, reason = pick_keeper(group, image_cache, args.model, args.llama_server)

    for path in group:
        is_keeper = (path.name == chosen)
        if is_keeper:
            shutil.copy2(path, output_dir / path.name)
            print(f"  cluster: {[p.name for p in group]}")
            print(f"  keeper: {path.name} — {reason}")
        write_row(writer, csvfile, filename=path.name, cluster_id=cluster_id,
                  blur_score=blur_scores[path], is_blurry=False, is_duplicate=True,
                  keeper=is_keeper, reason=reason if is_keeper else "duplicate")


def process_cluster(cluster, cluster_id, total, blur_scores, image_cache,
                    output_dir, writer, csvfile, args):
    """Process one burst cluster end to end."""
    print(f"\ncluster {cluster_id}/{total}: {len(cluster)} image(s)")

    sharp = [p for p in cluster if blur_scores[p] >= args.blur_threshold]
    blurry = [p for p in cluster if blur_scores[p] < args.blur_threshold]

    for path in blurry:
        print(f"  blurry: {path.name} (score: {blur_scores[path]:.1f})")
        write_row(writer, csvfile, filename=path.name, cluster_id=cluster_id,
                  blur_score=blur_scores[path], is_blurry=True, is_duplicate=False,
                  keeper=False, reason="below blur threshold")

    if not sharp:
        print(f"  all images blurry, skipping cluster")
        return

    if len(sharp) == 1:
        path = sharp[0]
        print(f"  singleton keeper: {path.name}")
        shutil.copy2(path, output_dir / path.name)
        write_row(writer, csvfile, filename=path.name, cluster_id=cluster_id,
                  blur_score=blur_scores[path], is_blurry=False, is_duplicate=False,
                  keeper=True, reason="singleton")
        return

    for group in find_duplicates(sharp, image_cache):
        process_duplicate_group(group, cluster_id, blur_scores, image_cache,
                                output_dir, writer, csvfile, args)


def process(args):
    input_dir, output_dir, report_path = resolve_paths(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"input:   {input_dir}")
    print(f"output:  {output_dir}")
    print(f"report:  {report_path}")
    print(f"blur threshold: {args.blur_threshold}")
    print(f"burst gap:      {args.burst_gap}s")
    print()

    print("scanning images...")
    images_ts = load_images(input_dir)
    if not images_ts:
        print("no supported images found in input directory", file=sys.stderr)
        sys.exit(1)
    print(f"found {len(images_ts)} images")

    print("loading images and computing blur scores...")
    image_cache, blur_scores = cache_and_score(images_ts)

    print("clustering bursts...")
    clusters = cluster_bursts(images_ts, args.burst_gap)
    print(f"found {len(clusters)} clusters")

    with open(report_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
            "filename", "cluster_id", "blur_score",
            "is_blurry", "is_duplicate", "keeper", "reason"
        ])
        writer.writeheader()
        csvfile.flush()

        for i, cluster in enumerate(clusters):
            process_cluster(cluster, i + 1, len(clusters), blur_scores,
                            image_cache, output_dir, writer, csvfile, args)

    print(f"\ndone. keepers copied to {output_dir}")
    print(f"report written to {report_path}")


if __name__ == "__main__":
    args = parse_args()
    process(args)
