#!/usr/bin/env python3
"""
Model download script for AInfer (Qwen 35B A3B / Tiel-Coder-35B-A3B-Genesis-Hermes).

Downloads:
1. Dequantized SafeTensors model & tokenizer from:
   symrex/Tiel-Coder-35B-A3B-Genesis-Hermes-GGUF-dequantized
2. GGUF reference model for llama.cpp from:
   LuffyTheFox/Tiel-Coder-35B-A3B-Genesis-Hermes-GGUF
"""

import argparse
import os
import sys
import time
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

DEQUANT_REPO = "symrex/Tiel-Coder-35B-A3B-Genesis-Hermes-GGUF-dequantized"
GGUF_REPO = "LuffyTheFox/Tiel-Coder-35B-A3B-Genesis-Hermes-GGUF"

METADATA_FILES = [
    "config.json",
    "generation_config.json",
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "README.md",
]


def download_file(repo_id: str, filename: str, local_dir: Path, max_retries: int = 5):
    print(f"[{repo_id}] Downloading {filename} -> {local_dir}...")
    local_path = local_dir / filename
    for attempt in range(1, max_retries + 1):
        try:
            downloaded = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(local_dir),
            )
            size_mb = Path(downloaded).stat().st_size / (1024 * 1024)
            print(f"[{repo_id}] Finished {filename} ({size_mb:.1f} MB)")
            return downloaded
        except Exception as e:
            print(f"[{repo_id}] Attempt {attempt}/{max_retries} failed for {filename}: {e}", file=sys.stderr)
            if attempt < max_retries:
                time.sleep(2 * attempt)
            else:
                raise


def get_safetensors_shards(api: HfApi, repo_id: str):
    info = api.model_info(repo_id)
    shards = [
        s.rfilename for s in info.siblings
        if s.rfilename.endswith(".safetensors")
    ]
    shards.sort()
    return shards


def get_gguf_files(api: HfApi, repo_id: str):
    info = api.model_info(repo_id)
    files = [
        s.rfilename for s in info.siblings
        if s.rfilename.endswith(".gguf")
    ]
    files.sort()
    return files


def main():
    parser = argparse.ArgumentParser(description="Download Qwen 35B A3B models for AInfer")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/Tiel-Coder-35B-A3B-Genesis-Hermes"),
        help="Local target directory for SafeTensors model",
    )
    parser.add_argument(
        "--gguf-dir",
        type=Path,
        default=Path("models/Tiel-Coder-35B-A3B-Genesis-Hermes"),
        help="Local target directory for GGUF reference model",
    )
    parser.add_argument(
        "--gguf-variant",
        type=str,
        default="Tiel-Coder-35B-A3B-Genesis-Hermes-APEX-Compact.gguf",
        help="GGUF filename to download (default: APEX-Compact.gguf; or 'all')",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Download only config and tokenizer metadata",
    )
    parser.add_argument(
        "--skip-gguf",
        action="store_true",
        help="Skip downloading GGUF files",
    )
    parser.add_argument(
        "--skip-safetensors",
        action="store_true",
        help="Skip downloading SafeTensors shard files",
    )
    args = parser.parse_args()

    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.gguf_dir.mkdir(parents=True, exist_ok=True)

    api = HfApi()

    # 1. Download metadata files first
    print("=== Step 1: Downloading metadata and tokenizer files ===")
    for fname in METADATA_FILES:
        try:
            download_file(DEQUANT_REPO, fname, args.model_dir)
        except Exception as e:
            print(f"Warning: Could not download {fname}: {e}", file=sys.stderr)

    if args.metadata_only:
        print("Metadata download complete.")
        return

    # 2. Download GGUF reference file
    if not args.skip_gguf:
        print("\n=== Step 2: Downloading GGUF reference model for llama.cpp ===")
        all_ggufs = get_gguf_files(api, GGUF_REPO)
        print("Available GGUF variants in repo:", all_ggufs)

        if args.gguf_variant == "all":
            targets = all_ggufs
        else:
            if args.gguf_variant in all_ggufs:
                targets = [args.gguf_variant]
            else:
                matches = [g for g in all_ggufs if args.gguf_variant.lower() in g.lower()]
                targets = matches if matches else all_ggufs[:1]

        for target in targets:
            download_file(GGUF_REPO, target, args.gguf_dir)

    # 3. Download SafeTensors shards
    if not args.skip_safetensors:
        print("\n=== Step 3: Downloading SafeTensors shards ===")
        shards = get_safetensors_shards(api, DEQUANT_REPO)
        print(f"Found {len(shards)} SafeTensors shards to download.")
        for idx, shard in enumerate(shards, 1):
            print(f"\n--- Shard {idx}/{len(shards)}: {shard} ---")
            download_file(DEQUANT_REPO, shard, args.model_dir)

    print("\n=== All downloads completed successfully! ===")


if __name__ == "__main__":
    main()
