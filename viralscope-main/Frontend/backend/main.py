"""
FastAPI backend server for ViralScope - YouTube Video Predictor.
"""
import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from backend.schemas import VideoFeatures, PredictionResponse, AgentRequest, AgentResponse
from backend.models.mock_model import predict as mock_predict
from backend.llm.agent_manager import run_agent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="ViralScope API",
    description="API for predicting YouTube video view performance",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://shimmering-happiness-production-a57e.up.railway.app"],  # Next.js frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "Viral Video Predictor API is running"}


@app.post("/predict", response_model=PredictionResponse)
async def predict_video(features: VideoFeatures) -> PredictionResponse:
    """
    Direct prediction endpoint accepting structured video metadata.
    Uses the trained RandomForest model for predictions and generates LLM analysis.
    
    REQUIRES: RandomForest.pkl file in backend/models/
    
    Args:
        features: Structured video features (including optional channel info)
        
    Returns:
        PredictionResponse with success/failure probabilities, predicted class, and LLM analysis
        
    Raises:
        HTTPException 404: If model file is not found
        HTTPException 500: If prediction fails
    """
    try:
        logger.info(f"Received prediction request for video: {features.title}")
        logger.info(f"Video features: category={features.category}, duration={features.duration}s, tags={features.tags}")
        
        # Get model prediction
        result = mock_predict(
            features,
            channel_subscribers=features.channel_subscribers,
            channel_total_videos=features.channel_total_videos,
            channel_views=features.channel_views,
            upload_day_of_week=features.upload_day_of_week
        )
        
        logger.info(f"Prediction completed: Class={result.predicted_class} ({'Success' if result.predicted_class == 1 else 'Failure'}), Success={result.success_probability:.3f}, Failure={result.failure_probability:.3f}")
        
        # Generate LLM analysis
        analysis = None
        try:
            from backend.llm.prediction_analyzer import analyze_prediction
            
            # Get model output details if available
            model_output = getattr(result, '_model_output', None)
            if model_output:
                features_dict = model_output.get('features_dict', {})
                analysis = analyze_prediction(
                    prediction_output=model_output,
                    features=features_dict,
                    conversation_history=None  # No conversation history for direct predict
                )
                logger.info(f"Generated LLM analysis: {analysis[:100]}...")
            else:
                # Fallback: reconstruct from result
                analysis = analyze_prediction(
                    prediction_output={
                        "prediction_class": result.predicted_class,
                        "success_probability": result.success_probability,
                        "failure_probability": result.failure_probability,
                        "features_dict": {},
                        "raw_features": features.model_dump()
                    },
                    features={},
                    conversation_history=None
                )
        except Exception as e:
            logger.warning(f"Failed to generate LLM analysis: {str(e)}")
            analysis = None
        
        # Add analysis to result
        if analysis:
            # Create new response with analysis
            result_dict = result.model_dump()
            result_dict["analysis"] = analysis
            return PredictionResponse(**result_dict)
        
        return result
    except FileNotFoundError as e:
        # Model file not found - return 404 with detailed message
        logger.error(f"Model file not found: {str(e)}")
        raise HTTPException(
            status_code=404,
            detail=f"Model file not found. {str(e)}"
        )
    except Exception as e:
        # Other errors - return 500 with error details
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/agent", response_model=AgentResponse)
async def agent_predict(request: AgentRequest) -> AgentResponse:
    """
    LLM agent endpoint accepting natural language query.
    
    Works in two modes:
    1. Conversational: Returns LLM response for general questions
    2. Prediction: Extracts features and calls prediction model when user requests review/prediction
    
    Args:
        request: AgentRequest with user_query string and optional conversation_history
        
    Returns:
        AgentResponse with either conversational response or prediction results
    """
    try:
        backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
        
        # Convert conversation history to dict format if provided
        conversation_history = None
        if request.conversation_history:
            conversation_history = [
                {"role": msg.role, "content": msg.content}
                for msg in request.conversation_history
            ]
        
        logger.info(f"Received agent request: {request.user_query[:100]}...")
        result = run_agent(request.user_query, backend_url, conversation_history)
        
        # Log agent response
        if result.get("conversational"):
            logger.info(f"Agent returned conversational response: {result.get('response', '')[:100]}...")
        elif result.get("requires_prediction"):
            logger.info(f"Agent returned prediction: Class={result.get('predicted_class', 0)}, Success={result.get('success_probability', 0):.3f}, Failure={result.get('failure_probability', 0):.3f}")
        
        # Handle both conversational and prediction responses
        return AgentResponse(**result)
    except Exception as e:
        logger.error(f"Agent processing failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent processing failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)



