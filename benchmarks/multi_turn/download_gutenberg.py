#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Download ebooks from Project Gutenberg for multi-turn conversation benchmarks.

Usage:
    python download_gutenberg.py --count 1000
    python download_gutenberg.py --count 1000 --start-id 1000 --update-json
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

# Script directory
SCRIPT_DIR = Path(__file__).parent


def download_book(book_id: int, output_dir: Path) -> tuple[int, bool, str]:
    """Download a single book from Project Gutenberg."""
    url = f"https://www.gutenberg.org/ebooks/{book_id}.txt.utf-8"
    output_path = output_dir / f"pg{book_id}.txt"

    if output_path.exists():
        return book_id, True, "already exists"

    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200 and len(response.content) > 1000:
            with open(output_path, "wb") as f:
                f.write(response.content)
            return book_id, True, f"{len(response.content) // 1024}KB"
        else:
            return book_id, False, f"HTTP {response.status_code}"
    except Exception as e:
        return book_id, False, str(e)[:50]


def update_json_config(output_dir: Path, json_path: Path) -> None:
    """Update generate_multi_turn.json with downloaded book filenames."""
    # Find all pg*.txt files
    txt_files = sorted(
        [f.name for f in output_dir.glob("pg*.txt") if f.stat().st_size > 1000],
        key=lambda x: int(x[2:-4]) if x[2:-4].isdigit() else 0,
    )

    if not txt_files:
        print("No valid text files found to add to config.")
        return

    # Read existing config
    with open(json_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Update text_files
    config["text_files"] = txt_files

    # Write back
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    print(f"\nUpdated {json_path}")
    print(f"  Added {len(txt_files)} text files to config")


def main():
    parser = argparse.ArgumentParser(
        description="Download Project Gutenberg ebooks for benchmarking"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1000,
        help="Number of books to download (default: 1000)",
    )
    parser.add_argument(
        "--start-id",
        type=int,
        default=1000,
        help="Starting book ID (default: 1000)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=10,
        help="Number of concurrent downloads (default: 10)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: same as this script)",
    )
    parser.add_argument(
        "--update-json",
        action="store_true",
        help="Update generate_multi_turn.json with downloaded files",
    )
    args = parser.parse_args()

    # Determine output directory
    output_dir = Path(args.output_dir) if args.output_dir else SCRIPT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    book_ids = list(range(args.start_id, args.start_id + args.count))

    print(f"Downloading {args.count} books (ID: {args.start_id}-{args.start_id + args.count - 1})")
    print(f"Output directory: {output_dir}")
    print(f"Concurrent downloads: {args.max_workers}")
    print()

    success_count = 0
    failed_count = 0
    skipped_count = 0

    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False
        print("(Install tqdm for progress bar: pip install tqdm)")

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(download_book, bid, output_dir): bid for bid in book_ids
        }

        if use_tqdm:
            iterator = tqdm(as_completed(futures), total=len(book_ids), desc="Downloading")
        else:
            iterator = as_completed(futures)
            total = len(book_ids)

        for i, future in enumerate(iterator):
            book_id, success, msg = future.result()
            if success:
                if "exists" in msg:
                    skipped_count += 1
                else:
                    success_count += 1
            else:
                failed_count += 1

            if not use_tqdm and (i + 1) % 100 == 0:
                print(f"  Progress: {i + 1}/{total} (success={success_count}, skipped={skipped_count}, failed={failed_count})")

    print(f"\nDownload complete!")
    print(f"  New downloads: {success_count}")
    print(f"  Already existed: {skipped_count}")
    print(f"  Failed: {failed_count}")

    # Count valid files
    valid_files = list(output_dir.glob("pg*.txt"))
    valid_files = [f for f in valid_files if f.stat().st_size > 1000]
    total_size = sum(f.stat().st_size for f in valid_files)
    print(f"  Total valid files: {len(valid_files)}")
    print(f"  Total size: {total_size // (1024 * 1024)} MB")

    # Update JSON config if requested
    if args.update_json:
        json_path = SCRIPT_DIR / "generate_multi_turn.json"
        if json_path.exists():
            update_json_config(output_dir, json_path)
        else:
            print(f"\nWarning: {json_path} not found, skipping JSON update")


if __name__ == "__main__":
    main()
