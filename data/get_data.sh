#!/usr/bin/env bash
# Downloads and unzips raw EEG data for subjects 1–5 from THINGS-EEG2

set -euo pipefail

DOWNLOAD_DIR="./data/raw"
mkdir -p "$DOWNLOAD_DIR"

# Figshare file IDs, one per subject (s1–s5) (ONLY 5 for now)
SUBJECTS=(01 02 03 04 05)
FILE_IDS=(33244238 33247340 33247355 33247361 33247376 )

for idx in "${!SUBJECTS[@]}"; do
    SUBJ="sub-${SUBJECTS[$idx]}"
    FILE_ID="${FILE_IDS[$idx]}"
    ZIP_PATH="$DOWNLOAD_DIR/${SUBJ}.zip"
    SUBJ_DIR="$DOWNLOAD_DIR/$SUBJ"

    echo "=========================================="
    echo "Downloading ${SUBJ} (Figshare ID: ${FILE_ID})..."
    echo "=========================================="

    curl -L --progress-bar \
        "https://ndownloader.figshare.com/files/${FILE_ID}" \
        -o "$ZIP_PATH"

    echo "Unzipping ${SUBJ}..."
    mkdir -p "$SUBJ_DIR"
    unzip -q "$ZIP_PATH" -d "$SUBJ_DIR"

    echo "Cleaning up zip..."
    rm "$ZIP_PATH"

    echo "Done: ${SUBJ}"
done

echo ""
echo "All 5 subjects downloaded and extracted to: $DOWNLOAD_DIR"