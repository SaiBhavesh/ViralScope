"""
LLM Agent Manager for extracting structured data from natural language.
Uses Groq API (OpenAI-compatible) to parse user queries and extract video metadata.
"""
import os
import json
import httpx
from typing import Dict, Any, Optional, List
from openai import OpenAI

from backend.schemas import VideoFeatures, PredictionResponse


# Initialize OpenAI client lazily
_client: Optional[OpenAI] = None

def get_openai_client() -> OpenAI:
    """Get or create OpenAI-compatible client instance (Groq)."""
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key or api_key == "your_groq_api_key_here":
            raise ValueError(
                "GROQ_API_KEY environment variable is not set. "
                "Please set it in your .env file."
            )
        # Groq uses OpenAI-compatible API
        _client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1"
        )
    return _client


def _call_llm(user_query: str) -> Dict[str, Any]:
    """
    Call Groq API (OpenAI-compatible) to extract structured video metadata from natural language.
    
    Args:
        user_query: Natural language description of the video
        
    Returns:
        Dictionary with extracted video features
    """
    system_prompt = """You are a video metadata extraction assistant. Extract structured information about a video from natural language descriptions.

You MUST respond with ONLY valid JSON in this exact format:
{
  "title": "string",
  "description": "string (optional, can be empty)",
  "tags": ["tag1", "tag2"],
  "category": "string (e.g., Entertainment, Education, Gaming, Music, etc.)",
  "duration": integer (in seconds),
  "upload_hour": integer (0-23, hour of day)
}

Rules:
- Extract the title from the description or infer a reasonable one
- Extract tags as a list of strings (split comma-separated tags if needed)
- Infer category from context (default to "Entertainment" if unclear)
- Extract duration in seconds (e.g., "2 minutes" = 120, "1 hour" = 3600)
- Extract upload hour if mentioned, otherwise use 12 (noon) as default
- Return ONLY the JSON object, no additional text or markdown formatting"""

    try:
        client = get_openai_client()
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # Groq model
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty response from LLM")
            
        # Parse JSON response
        parsed = json.loads(content)
        return parsed
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse LLM response as JSON: {str(e)}")
    except Exception as e:
        raise ValueError(f"LLM API call failed: {str(e)}")


def _validate_and_normalize(features_dict: Dict[str, Any]) -> VideoFeatures:
    """
    Validate and normalize extracted features.
    
    Args:
        features_dict: Raw dictionary from LLM
        
    Returns:
        Validated VideoFeatures object
    """
    # Normalize duration - ensure it's at least 1 second
    duration = features_dict.get("duration", 300)
    if not isinstance(duration, int):
        try:
            duration = int(float(duration))
        except (ValueError, TypeError):
            duration = 300
    # Ensure duration is at least 1 second (Pydantic constraint)
    duration = max(1, duration)
    
    # Normalize upload_hour
    upload_hour = features_dict.get("upload_hour", 12)
    if not isinstance(upload_hour, int):
        try:
            upload_hour = int(float(upload_hour))
        except (ValueError, TypeError):
            upload_hour = 12
    upload_hour = max(0, min(23, upload_hour))  # Clamp to 0-23
    
    # Normalize tags
    tags = features_dict.get("tags", [])
    if isinstance(tags, str):
        tags = [tag.strip() for tag in tags.split(",")]
    elif not isinstance(tags, list):
        tags = []
    
    # Normalize category
    category = features_dict.get("category", "Entertainment")
    if not isinstance(category, str):
        category = "Entertainment"
    
    # Normalize title
    title = features_dict.get("title", "")
    if not isinstance(title, str):
        title = str(title) if title else "Untitled Video"
    
    # Normalize description
    description = features_dict.get("description", "")
    if not isinstance(description, str):
        description = str(description) if description else ""
    
    return VideoFeatures(
        title=title,
        description=description,
        tags=tags,
        category=category,
        duration=duration,
        upload_hour=upload_hour
    )


def _call_model_direct(features: VideoFeatures) -> PredictionResponse:
    """
    Try to call the prediction model directly via import.
    
    Args:
        features: Video features to predict for
        
    Returns:
        PredictionResponse from model
    """
    try:
        from backend.models.mock_model import predict
        return predict(features)
    except ImportError:
        raise ImportError("Could not import model directly")


def _call_model_http(features: VideoFeatures, backend_url: str) -> PredictionResponse:
    """
    Call the prediction model via HTTP API.
    
    Args:
        features: Video features to predict for
        backend_url: Base URL of the backend API
        
    Returns:
        PredictionResponse from model
    """
    try:
        url = f"{backend_url}/predict"
        with httpx.Client(timeout=30.0) as http_client:
            response = http_client.post(
                url,
                json=features.model_dump(),
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return PredictionResponse(**response.json())
    except Exception as e:
        raise ValueError(f"HTTP call to prediction model failed: {str(e)}")


def _detect_prediction_request(user_query: str) -> bool:
    """
    Detect if the user is requesting a prediction/review.
    
    Args:
        user_query: User's natural language query
        
    Returns:
        True if user wants a prediction, False for conversational response
    """
    prediction_keywords = [
        "predict", "prediction", "review", "analyze", "estimate",
        "how many views", "viral potential", "performance",
        "will it go viral", "should I upload", "what are the chances"
    ]
    
    query_lower = user_query.lower()
    return any(keyword in query_lower for keyword in prediction_keywords)


def _conversational_response(user_query: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    """
    Generate a conversational LLM response without calling the prediction model.
    
    Args:
        user_query: User's natural language query
        
    Returns:
        Dictionary with conversational response
    """
    system_prompt = """You are a helpful assistant for YouTube creators. You provide advice, answer questions, and help with YouTube video-related topics. 
Be friendly, informative, and conversational. Always focus on YouTube specifically - mention YouTube features, YouTube best practices, YouTube algorithms, etc.

IMPORTANT: If the user asks about improving their YouTube video or making it better, and there's a previous prediction in the conversation history, reference that specific prediction (predicted YouTube views, confidence, video details like title, duration, category, upload hour) to provide contextual, personalized advice.

When referencing previous predictions:
- Mention the specific predicted YouTube views and confidence level
- Reference the YouTube video details (title, duration, category, tags, upload hour)
- Provide specific, actionable YouTube optimization suggestions based on those details
- Compare to YouTube best practices for that category and duration
- Mention YouTube-specific features like thumbnails, SEO, YouTube Shorts, etc.

If asked about YouTube video predictions or reviews without context, you can discuss general YouTube trends but note that specific predictions require analyzing YouTube video metadata."""

    try:
        client = get_openai_client()
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history if provided
        if conversation_history:
            for msg in conversation_history[-5:]:  # Last 5 messages for context
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role in ["user", "assistant"] and content:
                    messages.append({"role": role, "content": content})
        
        # Add current user query
        messages.append({"role": "user", "content": user_query})
        
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.7
        )
        
        content = response.choices[0].message.content
        if not content:
            content = "I'm here to help! Could you rephrase your question?"
        
        return {
            "conversational": True,
            "response": content,
            "requires_prediction": False
        }
    except Exception as e:
        return {
            "conversational": True,
            "response": f"I encountered an error: {str(e)}",
            "requires_prediction": False
        }


def run_agent(user_query: str, backend_url: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    """
    Main agent function: conversational LLM by default, only calls prediction model when requested.
    
    Flow:
    1. Check if user wants a prediction/review
    2. If yes: Extract features → Call prediction model
    3. If no: Return conversational LLM response
    
    Args:
        user_query: Natural language query from user
        backend_url: Base URL of the backend API (for HTTP fallback)
        
    Returns:
        Dictionary with either:
        - Conversational: {conversational: True, response: str, requires_prediction: False}
        - Prediction: {parsed_input: {...}, predicted_views: int, confidence: float, requires_prediction: True}
    """
    # Step 1: Detect if user wants a prediction
    wants_prediction = _detect_prediction_request(user_query)
    
    if not wants_prediction:
        # Return conversational response with conversation history
        return _conversational_response(user_query, conversation_history)
    
    # Step 2: User wants prediction - extract features and call model
    try:
        raw_features = _call_llm(user_query)
    except Exception as e:
        raise ValueError(f"Failed to extract features from query: {str(e)}")
    
    # Step 3: Validate and normalize features
    try:
        features = _validate_and_normalize(raw_features)
    except Exception as e:
        raise ValueError(f"Failed to validate features: {str(e)}")
    
    # Step 4: Call prediction model (hybrid approach)
    try:
        # Try direct import first (preferred path)
        prediction = _call_model_direct(features)
    except (FileNotFoundError, ImportError) as e:
        # If direct import fails due to missing model, try HTTP (might be in different container)
        # But if it's a real import error, re-raise
        try:
            prediction = _call_model_http(features, backend_url)
        except Exception as http_error:
            # Combine both error messages for clarity
            raise ValueError(
                f"Failed to get prediction:\n"
                f"  Direct import error: {str(e)}\n"
                f"  HTTP fallback error: {str(http_error)}\n\n"
                f"Please ensure RandomForest.pkl is available and the backend is running."
            ) from http_error
    except Exception as e:
        # Other errors from direct call
        raise ValueError(f"Model prediction failed: {str(e)}")
    
    # Step 5: Generate LLM analysis of the prediction
    analysis = None
    try:
        from backend.llm.prediction_analyzer import analyze_prediction
        
        # Get model output details if available
        model_output = getattr(prediction, '_model_output', None)
        if model_output:
            # Get features_dict from model output
            features_dict = model_output.get('features_dict', {})
            analysis = analyze_prediction(
                prediction_output=model_output,
                features=features_dict,
                conversation_history=conversation_history
            )
        else:
            # Fallback: reconstruct from prediction object
            # Note: This fallback may not have all feature details
            analysis = analyze_prediction(
                prediction_output={
                    "prediction_class": prediction.predicted_class,
                    "success_probability": prediction.success_probability,
                    "failure_probability": prediction.failure_probability,
                    "features_dict": {},
                    "raw_features": features.model_dump()
                },
                features={},
                conversation_history=conversation_history
            )
    except Exception as e:
        # If analysis fails, log but don't fail the whole request
        import logging
        logging.warning(f"Failed to generate prediction analysis: {str(e)}")
        analysis = None
    
    # Step 6: Combine results with analysis
    result = {
        "parsed_input": features.model_dump(),
        "success_probability": prediction.success_probability,
        "failure_probability": prediction.failure_probability,
        "predicted_class": prediction.predicted_class,
        "requires_prediction": True
    }
    
    # Add analysis if available
    if analysis:
        result["analysis"] = analysis
    
    return result



