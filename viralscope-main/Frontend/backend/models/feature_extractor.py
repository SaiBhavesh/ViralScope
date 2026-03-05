"""
Feature extraction module to transform user input into model-ready features.
Adapted from trendy-tube feature engineering pipeline.
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List
import pickle
import os

# Expected feature names from the RandomForest model
EXPECTED_FEATURES = [
    'channel_views', 'avg_views_per_video', 'avg_subs_per_video',
    'title_length', 'title_word_count', 'title_has_question', 
    'title_has_exclamation', 'title_uppercase_ratio',
    'description_length', 'description_word_count', 'has_description',
    'num_tags', 'duration_minutes', 'is_short_video', 'is_long_video',
    'channel_subscribers', 'channel_total_videos', 'subscriber_to_video_ratio',
    'day_0', 'day_1', 'day_2', 'day_3', 'day_4', 'day_5', 'day_6',
    'category_Autos & Vehicles', 'category_Comedy', 'category_Education',
    'category_Entertainment', 'category_Film & Animation', 'category_Gaming',
    'category_Howto & Style', 'category_Music', 'category_News & Politics',
    'category_Nonprofits & Activism', 'category_People & Blogs',
    'category_Pets & Animals', 'category_Science & Technology',
    'category_Sports', 'category_Travel & Events'
]


def extract_features_from_input(
    title: str,
    description: str,
    tags: List[str],
    category: str,
    duration: int,
    upload_hour: int,
    upload_day_of_week: int = None,
    channel_subscribers: int = 0,
    channel_total_videos: int = 0,
    channel_views: int = 0
) -> Dict[str, Any]:
    """
    Extract and engineer features from user input to match model format.
    
    Args:
        title: Video title
        description: Video description
        tags: List of tags
        category: Video category
        duration: Duration in seconds
        upload_hour: Hour of upload (0-23)
        upload_day_of_week: Day of week (0=Monday, 6=Sunday). If None, uses current day
        channel_subscribers: Channel subscriber count (default: 0)
        channel_total_videos: Channel total videos (default: 0)
        channel_views: Channel total views (default: 0)
        
    Returns:
        Dictionary with all engineered features
    """
    # Normalize inputs
    title = str(title) if title else ""
    description = str(description) if description else ""
    tags = tags if isinstance(tags, list) else []
    category = str(category) if category else "Entertainment"
    duration = max(1, int(duration))  # Ensure at least 1 second
    
    # === TITLE FEATURES ===
    title_length = len(title)
    title_word_count = len(title.split()) if title else 0
    title_has_question = 1 if '?' in title else 0
    title_has_exclamation = 1 if '!' in title else 0
    title_uppercase_ratio = (
        sum(1 for c in title if c.isupper()) / len(title) 
        if len(title) > 0 else 0
    )
    
    # === DESCRIPTION FEATURES ===
    description_length = len(description)
    description_word_count = len(description.split()) if description else 0
    has_description = 1 if description_length > 0 else 0
    
    # === TAGS FEATURES ===
    # Count unique tags (case-insensitive)
    unique_tags = set(tag.strip().lower() for tag in tags if tag.strip())
    num_tags = len(unique_tags)
    
    # === DURATION FEATURES ===
    duration_minutes = duration / 60.0
    is_short_video = 1 if duration_minutes < 5 else 0
    is_long_video = 1 if duration_minutes > 20 else 0
    
    # === UPLOAD TIMING FEATURES ===
    # Use provided day or current day
    if upload_day_of_week is None:
        upload_day_of_week = datetime.now().weekday()  # 0=Monday, 6=Sunday
    
    # One-hot encode day of week
    day_features = {f'day_{i}': 0 for i in range(7)}
    if 0 <= upload_day_of_week <= 6:
        day_features[f'day_{upload_day_of_week}'] = 1
    
    # === CHANNEL FEATURES ===
    # Use provided values or defaults
    channel_subscribers = max(0, int(channel_subscribers))
    channel_total_videos = max(0, int(channel_total_videos))
    channel_views = max(0, int(channel_views))
    
    # Calculate derived channel features
    avg_views_per_video = (
        channel_views / channel_total_videos 
        if channel_total_videos > 0 else 0
    )
    avg_subs_per_video = (
        channel_subscribers / channel_total_videos 
        if channel_total_videos > 0 else 0
    )
    subscriber_to_video_ratio = (
        channel_subscribers / (channel_total_videos + 1)
    )
    
    # === CATEGORY FEATURES ===
    # Normalize category name to match training data format
    category_mapping = {
        'autos': 'Autos & Vehicles',
        'vehicles': 'Autos & Vehicles',
        'comedy': 'Comedy',
        'education': 'Education',
        'entertainment': 'Entertainment',
        'film': 'Film & Animation',
        'animation': 'Film & Animation',
        'gaming': 'Gaming',
        'games': 'Gaming',
        'howto': 'Howto & Style',
        'style': 'Howto & Style',
        'music': 'Music',
        'news': 'News & Politics',
        'politics': 'News & Politics',
        'nonprofit': 'Nonprofits & Activism',
        'activism': 'Nonprofits & Activism',
        'people': 'People & Blogs',
        'blogs': 'People & Blogs',
        'pets': 'Pets & Animals',
        'animals': 'Pets & Animals',
        'science': 'Science & Technology',
        'technology': 'Science & Technology',
        'tech': 'Science & Technology',
        'sports': 'Sports',
        'travel': 'Travel & Events',
        'events': 'Travel & Events'
    }
    
    # Try to match category
    category_lower = category.lower().strip()
    matched_category = category
    for key, value in category_mapping.items():
        if key in category_lower:
            matched_category = value
            break
    
    # If no match, default to Entertainment
    if matched_category not in EXPECTED_FEATURES:
        matched_category = 'Entertainment'
    
    # One-hot encode category
    category_features = {f'category_{cat}': 0 for cat in [
        'Autos & Vehicles', 'Comedy', 'Education', 'Entertainment',
        'Film & Animation', 'Gaming', 'Howto & Style', 'Music',
        'News & Politics', 'Nonprofits & Activism', 'People & Blogs',
        'Pets & Animals', 'Science & Technology', 'Sports', 'Travel & Events'
    ]}
    
    category_key = f'category_{matched_category}'
    if category_key in category_features:
        category_features[category_key] = 1
    
    # === COMBINE ALL FEATURES ===
    features = {
        'channel_views': channel_views,
        'avg_views_per_video': avg_views_per_video,
        'avg_subs_per_video': avg_subs_per_video,
        'title_length': title_length,
        'title_word_count': title_word_count,
        'title_has_question': title_has_question,
        'title_has_exclamation': title_has_exclamation,
        'title_uppercase_ratio': title_uppercase_ratio,
        'description_length': description_length,
        'description_word_count': description_word_count,
        'has_description': has_description,
        'num_tags': num_tags,
        'duration_minutes': duration_minutes,
        'is_short_video': is_short_video,
        'is_long_video': is_long_video,
        'channel_subscribers': channel_subscribers,
        'channel_total_videos': channel_total_videos,
        'subscriber_to_video_ratio': subscriber_to_video_ratio,
        **day_features,
        **category_features
    }
    
    return features


def prepare_features_for_model(
    features_dict: Dict[str, Any],
    scaler_path: str = None,
    model_path: str = None
) -> np.ndarray:
    """
    Prepare features for model prediction.
    
    Note: RandomForest is tree-based and doesn't require feature scaling.
    The scaler parameter is kept for compatibility but is not used.
    """
    """
    Prepare features for model prediction: scale and ensure correct order.
    
    Args:
        features_dict: Dictionary of extracted features
        scaler_path: Path to scaler.pkl (if None, uses default location)
        model_path: Path to model.pkl (if None, uses default location)
        
    Returns:
        numpy array of features in correct order for model
    """
    # Load model to get feature order
    if model_path is None:
        model_path = os.path.join(os.path.dirname(__file__), 'RandomForest.pkl')
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Get expected feature order from model
    if hasattr(model, 'feature_names_in_'):
        feature_order = list(model.feature_names_in_)
    else:
        # Fallback to expected features if model doesn't have feature_names_in_
        feature_order = EXPECTED_FEATURES
    
    # Create DataFrame with single row
    df = pd.DataFrame([features_dict])
    
    # Ensure all expected features exist (fill missing with 0)
    for feature in feature_order:
        if feature not in df.columns:
            df[feature] = 0
    
    # Reorder columns to match model's expected order
    df = df[feature_order]
    
    # Note: RandomForest is tree-based and doesn't require feature scaling
    # If a scaler was used during training, it would be needed, but since
    # scaler.pkl doesn't exist, we skip scaling (RandomForest works fine without it)
    
    # Convert to numpy array
    feature_array = df.values[0]
    
    # Handle any NaN or inf values
    feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=0.0, neginf=0.0)
    
    return feature_array.reshape(1, -1)  # Reshape for single prediction

