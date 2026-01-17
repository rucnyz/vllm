#!/bin/bash

# Copy grid search results from source to destination, preserving directory structure
# Usage: ./copy_grid_search.sh <source_dir> <dest_dir>
# Example: ./copy_grid_search.sh grid_search_v0 grid_search_20260116_004326

set -e

if [ $# -ne 2 ]; then
    echo "Usage: $0 <source_dir> <dest_dir>"
    echo "Example: $0 grid_search_v0 grid_search_20260116_004326"
    exit 1
fi

SOURCE_DIR="$1"
DEST_DIR="$2"

# Check if source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: Source directory '$SOURCE_DIR' does not exist"
    exit 1
fi

# Check if destination directory exists
if [ ! -d "$DEST_DIR" ]; then
    echo "Error: Destination directory '$DEST_DIR' does not exist"
    exit 1
fi

echo "Copying from: $SOURCE_DIR"
echo "Copying to: $DEST_DIR"
echo ""

# Find all leaf directories (tb*/bs*/in*_out*/) in source
# These are directories matching the pattern tb*/bs*/in*_out*/
count=0
for tb_dir in "$SOURCE_DIR"/tb*/; do
    if [ ! -d "$tb_dir" ]; then
        continue
    fi
    tb_name=$(basename "$tb_dir")

    for bs_dir in "$tb_dir"/bs*/; do
        if [ ! -d "$bs_dir" ]; then
            continue
        fi
        bs_name=$(basename "$bs_dir")

        for in_out_dir in "$bs_dir"/in*_out*/; do
            if [ ! -d "$in_out_dir" ]; then
                continue
            fi
            in_out_name=$(basename "$in_out_dir")

            rel_path="$tb_name/$bs_name/$in_out_name"
            src_path="$SOURCE_DIR/$rel_path"
            dst_path="$DEST_DIR/$rel_path"

            # Check if source has any files to copy
            file_count=$(find "$src_path" -type f 2>/dev/null | wc -l)
            if [ "$file_count" -eq 0 ]; then
                echo "Skipping (no files): $rel_path"
                continue
            fi

            # Create destination directory if it doesn't exist
            mkdir -p "$dst_path"

            # Copy all contents from source to destination
            # Using cp -r to copy recursively, -n to not overwrite existing files
            echo "Copying: $rel_path ($file_count files)"
            cp -rn "$src_path"/* "$dst_path"/ 2>/dev/null || true

            count=$((count + 1))
        done
    done
done

echo ""
echo "Done! Copied $count directories."
