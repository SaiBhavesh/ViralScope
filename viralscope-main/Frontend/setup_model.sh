#!/bin/bash
# Script to copy RandomForest model files from trendy-tube-main to CS513 project

echo "Setting up RandomForest model files..."
echo "======================================"

TARGET_DIR="backend/models"

# Check if target directory exists
if [ ! -d "$TARGET_DIR" ]; then
    echo "Creating target directory: $TARGET_DIR"
    mkdir -p "$TARGET_DIR"
fi

# Try multiple possible source locations
POSSIBLE_SOURCES=(
    "../trendy-tube-main/models"
    "../../trendy-tube-main/models"
    "$HOME/Downloads/trendy-tube-main/models"
    "/Users/jashmehta/Downloads/trendy-tube-main/models"
)

SOURCE_DIR=""
for dir in "${POSSIBLE_SOURCES[@]}"; do
    if [ -d "$dir" ] && [ -f "$dir/RandomForest.pkl" ]; then
        SOURCE_DIR="$dir"
        echo "Found model files at: $SOURCE_DIR"
        break
    fi
done

# If not found, ask user
if [ -z "$SOURCE_DIR" ]; then
    echo ""
    echo "⚠️  Model files not found in common locations."
    echo ""
    echo "Please provide the path to trendy-tube-main/models directory:"
    echo "  (e.g., /path/to/trendy-tube-main/models or ../trendy-tube-main/models)"
    read -p "Enter path: " user_path
    
    if [ -d "$user_path" ] && [ -f "$user_path/RandomForest.pkl" ]; then
        SOURCE_DIR="$user_path"
    else
        echo "✗ Invalid path or files not found"
        echo ""
        echo "Please copy the files manually:"
        echo "  cp <path-to-trendy-tube-main>/models/RandomForest.pkl $TARGET_DIR/"
        echo "  cp <path-to-trendy-tube-main>/models/scaler.pkl $TARGET_DIR/"
        exit 1
    fi
fi

# Copy model files
echo ""
echo "Copying model files from: $SOURCE_DIR"
echo ""

if [ -f "$SOURCE_DIR/RandomForest.pkl" ]; then
    cp "$SOURCE_DIR/RandomForest.pkl" "$TARGET_DIR/"
    echo "✓ Copied RandomForest.pkl"
else
    echo "✗ RandomForest.pkl not found in $SOURCE_DIR"
fi

# Scaler is optional - RandomForest doesn't require it
if [ -f "$SOURCE_DIR/scaler.pkl" ]; then
    cp "$SOURCE_DIR/scaler.pkl" "$TARGET_DIR/"
    echo "✓ Copied scaler.pkl (optional)"
else
    echo "ℹ️  scaler.pkl not found (optional - RandomForest doesn't require scaling)"
fi

echo ""
echo "Verifying files..."
if [ -f "$TARGET_DIR/RandomForest.pkl" ]; then
    echo "✓ Model setup complete!"
    echo ""
    echo "Files in $TARGET_DIR:"
    ls -lh "$TARGET_DIR"/*.pkl
    echo ""
    echo "You can now use the real RandomForest model for predictions!"
else
    echo "✗ RandomForest.pkl is missing (required)"
    echo ""
    echo "Please copy it manually:"
    echo "  cp $SOURCE_DIR/RandomForest.pkl $TARGET_DIR/"
    echo ""
    echo "Note: scaler.pkl is optional - RandomForest doesn't require scaling"
    exit 1
fi

