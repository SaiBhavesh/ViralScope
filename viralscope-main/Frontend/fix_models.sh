#!/bin/bash
# Quick script to copy model files and restart Docker

echo "🔧 Fixing model files for Docker..."

# Copy model files
if [ -f "../trendy-tube-main/models/RandomForest.pkl" ]; then
    mkdir -p backend/models
    cp -v ../trendy-tube-main/models/RandomForest.pkl backend/models/
    # Scaler is optional - copy if it exists
    if [ -f "../trendy-tube-main/models/scaler.pkl" ]; then
        cp -v ../trendy-tube-main/models/scaler.pkl backend/models/
    fi
    echo "✓ Model file copied"
    
    # Verify required file
    if [ -f "backend/models/RandomForest.pkl" ]; then
        echo "✓ Files verified"
        ls -lh backend/models/*.pkl
        echo ""
        echo "🔄 Restarting Docker backend to load models..."
        docker-compose restart backend
        echo ""
        echo "✓ Done! Check the logs: docker-compose logs backend"
    else
        echo "✗ Files not found after copy"
    fi
else
    echo "✗ Model files not found at ../trendy-tube-main/models/"
    echo "Please provide the correct path to trendy-tube-main/models/"
fi

