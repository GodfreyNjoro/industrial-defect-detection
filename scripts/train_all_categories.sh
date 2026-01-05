#!/bin/bash

# Script to train models on all MVTec AD categories

set -e

# Categories in MVTec AD dataset
CATEGORIES=(
    "bottle"
    "cable"
    "capsule"
    "carpet"
    "grid"
    "hazelnut"
    "leather"
    "metal_nut"
    "pill"
    "screw"
    "tile"
    "toothbrush"
    "transistor"
    "wood"
    "zipper"
)

echo "========================================"
echo "Training models for all categories"
echo "========================================"
echo ""

# Configuration
CONFIG="configs/config.yaml"
EPOCHS=100
BATCH_SIZE=16

for category in "${CATEGORIES[@]}"; do
    echo "========================================"
    echo "Training category: $category"
    echo "========================================"
    echo ""
    
    python train.py \
        --config "$CONFIG" \
        --category "$category" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE"
    
    echo ""
    echo "Completed training for $category"
    echo ""
done

echo "========================================"
echo "All training completed!"
echo "========================================"
