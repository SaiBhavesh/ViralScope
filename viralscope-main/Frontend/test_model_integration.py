"""
Test script to verify RandomForest model integration.
Run this to test if the model loads and makes predictions correctly.
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from backend.schemas import VideoFeatures, PredictionResponse
from backend.models.mock_model import predict

def test_model_integration():
    """Test the model integration with sample data."""
    print("=" * 60)
    print("Testing RandomForest Model Integration")
    print("=" * 60)
    
    # Create sample video features
    sample_features = VideoFeatures(
        title="Amazing Cat Video - You Won't Believe This!",
        description="Watch this incredible cat do amazing tricks. Subscribe for more!",
        tags=["cat", "funny", "pets", "viral"],
        category="Entertainment",
        duration=180,  # 3 minutes
        upload_hour=14,  # 2 PM
        channel_subscribers=10000,
        channel_total_videos=50,
        channel_views=500000,
        upload_day_of_week=2  # Wednesday
    )
    
    print("\nSample Video Features:")
    print(f"  Title: {sample_features.title}")
    print(f"  Category: {sample_features.category}")
    print(f"  Duration: {sample_features.duration} seconds")
    print(f"  Tags: {sample_features.tags}")
    print(f"  Channel: {sample_features.channel_subscribers} subscribers, {sample_features.channel_total_videos} videos")
    
    try:
        print("\n" + "-" * 60)
        print("Making prediction...")
        print("-" * 60)
        
        result = predict(sample_features)
        
        print("\n✓ Prediction successful!")
        print(f"\nResults:")
        print(f"  Predicted Views: {result.predicted_views:,}")
        print(f"  Confidence: {result.confidence:.3f}")
        
        print("\n" + "=" * 60)
        print("✓ Model integration test PASSED")
        print("=" * 60)
        return True
        
    except FileNotFoundError as e:
        print(f"\n✗ Model file not found: {e}")
        print("\nPlease copy RandomForest.pkl and scaler.pkl to backend/models/")
        print("=" * 60)
        return False
    except Exception as e:
        print(f"\n✗ Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 60)
        return False

if __name__ == "__main__":
    success = test_model_integration()
    sys.exit(0 if success else 1)

