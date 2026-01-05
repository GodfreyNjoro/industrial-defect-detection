#!/bin/bash

# Script to download MVTec AD dataset

set -e  # Exit on error

echo "========================================"
echo "MVTec AD Dataset Download Script"
echo "========================================"
echo ""

# Create data directory
DATA_DIR="data/raw"
mkdir -p "$DATA_DIR"

echo "Data directory: $DATA_DIR"
echo ""

# Download dataset
DATASET_URL="https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads/mvtec_anomaly_detection.tar.xz"
DATASET_FILE="$DATA_DIR/mvtec_anomaly_detection.tar.xz"

if [ -f "$DATASET_FILE" ]; then
    echo "Dataset archive already exists. Skipping download."
else
    echo "Downloading MVTec AD dataset..."
    echo "URL: $DATASET_URL"
    echo "This may take a while (approx. 4.3 GB)..."
    echo ""
    
    wget -O "$DATASET_FILE" "$DATASET_URL" --progress=bar:force:noscroll
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "Download completed successfully!"
    else
        echo ""
        echo "Error: Download failed!"
        exit 1
    fi
fi

echo ""

# Extract dataset
if [ -d "$DATA_DIR/mvtec_ad" ]; then
    echo "Dataset already extracted. Skipping extraction."
else
    echo "Extracting dataset..."
    tar -xf "$DATASET_FILE" -C "$DATA_DIR"
    
    if [ $? -eq 0 ]; then
        echo "Extraction completed successfully!"
    else
        echo "Error: Extraction failed!"
        exit 1
    fi
fi

echo ""

# Print dataset info
echo "========================================"
echo "Dataset Information"
echo "========================================"
echo ""
echo "Dataset location: $DATA_DIR/mvtec_ad"
echo ""
echo "Categories:"
ls -1 "$DATA_DIR/mvtec_ad" 2>/dev/null || echo "No categories found. Check extraction."

echo ""
echo "========================================"
echo "Download and extraction complete!"
echo "========================================"
echo ""
echo "You can now start training:"
echo "  python train.py --config configs/config.yaml"
echo ""
