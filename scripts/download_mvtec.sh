#!/bin/bash
# Download and extract MVTec AD dataset to /home/phd/datasets/mvtec_vibration/

set -e  # Exit on any error

DEST_ROOT="/home/phd/datasets"
DATASET_NAME="mvtec_vibration"
DEST_DIR="$DEST_ROOT/$DATASET_NAME"

# Create destination
mkdir -p "$DEST_DIR"

# URLs (official MVTec AD)
MVTEC_URL="https://www.mvtec.com/company/mvtec-ad"

echo "  MVTec AD requires manual download due to license restrictions."
echo "  Please visit: $MVTEC_URL"
echo "  Download 'mvtec_ad.zip' and place it in $DEST_DIR/"
echo ""

read -p "Have you placed 'mvtec_ad.zip' in $DEST_DIR? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo " Aborted. Please download the dataset manually."
    exit 1
fi

# Verify zip exists
if [ ! -f "$DEST_DIR/mvtec_ad.zip" ]; then
    echo " mvtec_ad.zip not found in $DEST_DIR"
    exit 1
fi

echo " Extracting MVTec AD..."
unzip -q "$DEST_DIR/mvtec_ad.zip" -d "$DEST_DIR/"

# Optional: create symbolic link for code compatibility
if [ ! -L "$DEST_DIR/images" ]; then
    ln -s "$DEST_DIR/mvtec_ad" "$DEST_DIR/images"
fi

echo " MVTec AD ready at $DEST_DIR"