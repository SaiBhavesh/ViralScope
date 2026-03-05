# Copy Model Files - Quick Guide

## Quick Copy Command

If `trendy-tube-main` is in the same parent directory as `CS513-FinalProject`:

```bash
cd /Users/jashmehta/Downloads/CS513-FinalProject
cp ../trendy-tube-main/models/RandomForest.pkl backend/models/
# Note: scaler.pkl is optional - RandomForest doesn't require scaling
```

## Or Use the Setup Script

```bash
cd /Users/jashmehta/Downloads/CS513-FinalProject
./setup_model.sh
```

The script will:
1. Try to find the model files automatically
2. Ask you for the path if not found
3. Copy the files to the correct location

## Verify Files Are Copied

```bash
ls -lh backend/models/*.pkl
```

You should see:
- `RandomForest.pkl` (required)
- `scaler.pkl` (optional - not needed for RandomForest)

## After Copying

Restart your backend server. The model will be loaded automatically on first use.

```bash
# If using Docker
docker-compose restart backend

# If running locally
# Stop and restart your backend server
```

## Note

The app works without model files (uses mock predictions), but for real predictions, you need these files.

