"""
RandomForest model for viral video prediction.
Uses the trained RandomForest model from trendy-tube.

REQUIRES: RandomForest.pkl file in backend/models/
"""
import os
import pickle
import numpy as np
import logging
from typing import Optional
from backend.schemas import VideoFeatures, PredictionResponse
from backend.models.feature_extractor import (
    extract_features_from_input,
    prepare_features_for_model
)

# Set up logger
logger = logging.getLogger(__name__)


# Load model lazily
_model = None
_model_path = None


def _load_model():
    """Load the RandomForest model lazily."""
    global _model, _model_path
    
    if _model is None:
        # Try to find model file - check multiple possible locations
        base_dir = os.path.dirname(__file__)
        possible_paths = [
            os.path.join(base_dir, 'RandomForest.pkl'),  # Same directory as this file
            os.path.join(base_dir, 'models', 'RandomForest.pkl'),  # Nested models dir
            '/app/backend/models/RandomForest.pkl',  # Docker container path
            'backend/models/RandomForest.pkl',  # Relative path
            os.path.join(os.path.dirname(os.path.dirname(base_dir)), 'backend', 'models', 'RandomForest.pkl'),  # From project root
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                _model_path = path
                try:
                    with open(path, 'rb') as f:
                        _model = pickle.load(f)
                    print(f"✓ Loaded RandomForest model from: {path}")
                    break
                except Exception as e:
                    print(f"Warning: Failed to load model from {path}: {e}")
                    continue
        
        if _model is None:
            raise FileNotFoundError(
                "RandomForest.pkl not found. Please copy it to backend/models/"
            )
    
    return _model


# Scaler removed - RandomForest doesn't require feature scaling


def predict(
    features: VideoFeatures,
    channel_subscribers: Optional[int] = None,
    channel_total_videos: Optional[int] = None,
    channel_views: Optional[int] = None,
    upload_day_of_week: Optional[int] = None
) -> PredictionResponse:
    """
    Generate predictions using the trained RandomForest model.
    
    Args:
        features: Video features to predict for
        channel_subscribers: Optional channel subscriber count (default: 0)
        channel_total_videos: Optional channel total videos (default: 0)
        channel_views: Optional channel total views (default: 0)
        upload_day_of_week: Optional day of week 0-6 (default: current day)
        
    Returns:
        PredictionResponse with predicted success probability and confidence
    """
    try:
        # Load model
        model = _load_model()
        
        # Extract features from user input
        features_dict = extract_features_from_input(
            title=features.title,
            description=features.description or "",
            tags=features.tags,
            category=features.category,
            duration=features.duration,
            upload_hour=features.upload_hour,
            upload_day_of_week=upload_day_of_week,
            channel_subscribers=channel_subscribers or 0,
            channel_total_videos=channel_total_videos or 0,
            channel_views=channel_views or 0
        )
        
        # Log extracted features
        logger.info("=" * 80)
        logger.info("MODEL PREDICTION REQUEST")
        logger.info("=" * 80)
        logger.info(f"Input Features:")
        logger.info(f"  Title: {features.title}")
        logger.info(f"  Description: {features.description or '(empty)'}")
        logger.info(f"  Tags: {features.tags}")
        logger.info(f"  Category: {features.category}")
        logger.info(f"  Duration: {features.duration} seconds")
        logger.info(f"  Upload Hour: {features.upload_hour}")
        logger.info(f"  Channel Subscribers: {channel_subscribers or 0}")
        logger.info(f"  Channel Total Videos: {channel_total_videos or 0}")
        logger.info(f"  Channel Views: {channel_views or 0}")
        logger.info(f"  Upload Day of Week: {upload_day_of_week}")
        
        # Prepare features for model (order and format)
        feature_array = prepare_features_for_model(
            features_dict,
            scaler_path=None,  # No scaler needed for RandomForest
            model_path=_model_path
        )
        
        # Log feature array details
        logger.info(f"\nFeature Array Shape: {feature_array.shape}")
        logger.info(f"Feature Array (first 20 values): {feature_array[0][:20]}")
        
        # Make prediction
        # Model predicts binary classification (0 = not successful, 1 = successful)
        prediction_proba = model.predict_proba(feature_array)[0]
        prediction_class = model.predict(feature_array)[0]
        
        # Get probability of success (class 1) and failure (class 0)
        success_probability = float(prediction_proba[1])
        failure_probability = float(prediction_proba[0])
        
        # Log complete raw model output
        logger.info("\n" + "=" * 80)
        logger.info("RAW MODEL OUTPUT")
        logger.info("=" * 80)
        logger.info(f"Predicted Class: {prediction_class} (0=Failure, 1=Success)")
        logger.info(f"Prediction Probabilities (raw): {prediction_proba}")
        logger.info(f"  - Failure (Class 0): {failure_probability:.6f} ({failure_probability*100:.2f}%)")
        logger.info(f"  - Success (Class 1): {success_probability:.6f} ({success_probability*100:.2f}%)")
        
        # Create response object with probabilities only
        result = PredictionResponse(
            success_probability=round(success_probability, 6),
            failure_probability=round(failure_probability, 6),
            predicted_class=int(prediction_class)
        )
        
        # Log complete processed response
        logger.info("\n" + "=" * 80)
        logger.info("PROCESSED MODEL RESPONSE")
        logger.info("=" * 80)
        logger.info(f"Predicted Class: {prediction_class} ({'Success' if prediction_class == 1 else 'Failure'})")
        logger.info(f"Success Probability: {success_probability:.6f} ({success_probability*100:.2f}%)")
        logger.info(f"Failure Probability: {failure_probability:.6f} ({failure_probability*100:.2f}%)")
        logger.info(f"\nComplete Response Object:")
        logger.info(f"  {result.model_dump_json(indent=2)}")
        logger.info("=" * 80 + "\n")
        
        # Store additional info for LLM analysis (attach to result object)
        result._model_output = {
            "prediction_class": int(prediction_class),
            "success_probability": success_probability,
            "failure_probability": failure_probability,
            "features_dict": features_dict,
            "raw_features": features.model_dump()
        }
        
        return result
        
    except FileNotFoundError as e:
        # Re-raise with clear error message
        raise FileNotFoundError(
            f"Model file not found. Cannot make predictions.\n\n"
            f"Error details: {str(e)}\n\n"
            f"Please ensure RandomForest.pkl is in backend/models/ directory."
        ) from e
    except Exception as e:
        # Re-raise with context
        raise RuntimeError(
            f"Failed to make prediction with RandomForest model: {str(e)}\n\n"
            f"Model path: {_model_path}\n"
            f"Please check that the model file is valid and all dependencies are installed."
        ) from e



