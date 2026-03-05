# RandomForest Model Integration

## Overview

The RandomForest model from `trendy-tube-main` has been integrated into the ViralScope application. The model predicts YouTube video success based on video metadata and channel characteristics.

## Files Added/Modified

### New Files

1. **`backend/models/feature_extractor.py`**
   - Feature engineering module that transforms user input into model-ready features
   - Extracts title, description, tags, duration, category, and channel features
   - Handles one-hot encoding for day of week and categories
   - Applies scaling using the trained scaler

2. **`test_model_integration.py`**
   - Test script to verify model integration
   - Run with: `python3 test_model_integration.py`

### Modified Files

1. **`backend/models/mock_model.py`**
   - Replaced mock predictions with real RandomForest model
   - Loads model and scaler lazily
   - Falls back to mock predictions if model files are not found
   - Supports optional channel features

2. **`backend/schemas.py`**
   - Added optional channel features to `VideoFeatures`:
     - `channel_subscribers` (default: 0)
     - `channel_total_videos` (default: 0)
     - `channel_views` (default: 0)
     - `upload_day_of_week` (default: None, uses current day)

3. **`backend/main.py`**
   - Updated `/predict` endpoint to pass channel features to model

4. **`backend/requirements.txt`**
   - Added `pandas>=2.0.0` and `scikit-learn>=1.3.0` dependencies

## Model Files Required

Copy these files from `trendy-tube-main/models/` to `backend/models/`:

```bash
cp trendy-tube-main/models/RandomForest.pkl backend/models/
cp trendy-tube-main/models/scaler.pkl backend/models/
```

## Model Features

The model expects 40 features:

### Video Metadata (15 features)
- Title: length, word count, has question, has exclamation, uppercase ratio
- Description: length, word count, has description
- Tags: count
- Duration: minutes, is short (<5 min), is long (>20 min)

### Channel Features (6 features)
- channel_views
- avg_views_per_video
- avg_subs_per_video
- channel_subscribers
- channel_total_videos
- subscriber_to_video_ratio

### Temporal Features (7 features)
- Day of week (one-hot encoded: day_0 through day_6)

### Category Features (15 features)
- Category (one-hot encoded: category_Entertainment, category_Gaming, etc.)

## Usage

### Direct API Call

```python
from backend.schemas import VideoFeatures
from backend.models.mock_model import predict

features = VideoFeatures(
    title="My Amazing Video",
    description="This is a great video",
    tags=["funny", "viral"],
    category="Entertainment",
    duration=180,
    upload_hour=14,
    channel_subscribers=10000,  # Optional
    channel_total_videos=50,    # Optional
    channel_views=500000         # Optional
)

result = predict(features)
print(f"Predicted views: {result.predicted_views}")
print(f"Confidence: {result.confidence}")
```

### Via LLM Agent

The LLM agent automatically extracts features from natural language and calls the model:

```
User: "I'm uploading a funny cat video, 3 minutes long, in Entertainment category. Predict its performance."
```

The agent will:
1. Extract features from the query
2. Call the RandomForest model
3. Return predicted views and confidence

## Model Output

The model returns:
- **predicted_views**: Estimated view count (1K - 500K range)
- **confidence**: Model confidence score (0.0 - 1.0)

The prediction is based on the model's probability of success (class 1), mapped to a view count range.

## Fallback Behavior

If model files are not found, the system falls back to mock predictions to ensure the API remains functional during development.

## Testing

Run the test script:

```bash
python3 test_model_integration.py
```

This will verify:
- Model file loading
- Feature extraction
- Prediction generation

## Notes

- Channel features default to 0 if not provided
- Day of week defaults to current day if not specified
- Category is automatically mapped to match training data format
- All features are scaled using the trained StandardScaler before prediction

