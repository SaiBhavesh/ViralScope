"""
LLM-based analysis of model predictions.
Takes model output and generates actionable feedback for video creators.
"""
import logging
from typing import Dict, Any, Optional, List
from openai import OpenAI
import os

from backend.llm.agent_manager import get_openai_client

logger = logging.getLogger(__name__)


def analyze_prediction(
    prediction_output: Dict[str, Any],
    features: Dict[str, Any],
    conversation_history: Optional[List[Dict[str, str]]] = None
) -> str:
    """
    Use LLM to analyze model prediction and generate actionable feedback.
    
    Args:
        prediction_output: Dictionary with model output including:
            - prediction_class: 0 or 1
            - success_probability: float (0-1)
            - not_success_probability: float (0-1)
            - features_dict: extracted features
            - raw_features: original video features
        features: Original VideoFeatures object
        conversation_history: Optional conversation history for context
        
    Returns:
        Analysis text with actionable feedback
    """
    try:
        client = get_openai_client()
        
        # Extract model output details
        prediction_class = prediction_output.get("prediction_class", 0)
        success_prob = prediction_output.get("success_probability", 0.0)
        failure_prob = prediction_output.get("failure_probability", prediction_output.get("not_success_probability", 0.0))
        features_dict = prediction_output.get("features_dict", {})
        raw_features = prediction_output.get("raw_features", {})
        
        # Determine tone based on success probability
        is_high_success = success_prob >= 0.8
        is_low_success = success_prob < 0.5
        
        # Build system prompt
        if is_high_success:
            tone_instruction = f"The model predicts HIGH SUCCESS ({success_prob*100:.1f}% chance of success, {failure_prob*100:.1f}% chance of failure). Provide positive, encouraging feedback highlighting what's working well, while still offering minor optimization suggestions."
        elif is_low_success:
            tone_instruction = f"The model predicts LOW SUCCESS ({success_prob*100:.1f}% chance of success, {failure_prob*100:.1f}% chance of failure). Provide constructive, actionable feedback on how to improve. Be specific about what needs to change."
        else:
            tone_instruction = f"The model predicts MODERATE SUCCESS ({success_prob*100:.1f}% chance of success, {failure_prob*100:.1f}% chance of failure). Provide balanced feedback with both positive aspects and specific improvement areas."
        
        system_prompt = f"""You are an expert YouTube video analyst. Your job is to interpret machine learning model predictions and provide actionable feedback to video creators.

{tone_instruction}

You will receive:
1. Model prediction results (success probability, predicted class)
2. Video features and metadata
3. Extracted feature details (title length, word count, tags, etc.)

Your task:
- Interpret the model's prediction in plain language
- Analyze each feature and explain its impact
- Provide SPECIFIC, ACTIONABLE recommendations
- Reference the actual video details (title, tags, description, etc.)
- Be constructive and helpful

IMPORTANT GUIDELINES:
- If success probability is <50%: Start with "We feel this video may not be successful with a {failure_prob*100:.1f}% chance of failure ({success_prob*100:.1f}% chance of success). Here's how to improve it:"
- If success probability is >=80%: Start with "Great news! The model predicts strong success potential ({success_prob*100:.1f}% chance of success, {failure_prob*100:.1f}% chance of failure). Here's what's working well:"
- Analyze specific features:
  * Title: length, word count, question marks, exclamation marks, uppercase ratio
  * Description: length, word count, presence
  * Tags: count, suggest if too generic or too few
  * Duration: whether it's short (<5min), long (>20min), or optimal
  * Category: relevance and competition
  * Upload timing: hour and day of week
  * Channel metrics: subscribers, videos, views (if provided)
- Be specific: "Your tags like 'cat', 'funny' are too generic" not "Your tags could be better"
- Provide concrete examples when possible
- Keep response concise but comprehensive (2-4 paragraphs)

Format your response as natural, conversational advice."""

        # Build user message with all context
        user_message = f"""Model Prediction Results:
- Predicted Class: {prediction_class} (0 = Failure, 1 = Success)
- Success Probability: {success_prob:.4f} ({success_prob*100:.2f}%)
- Failure Probability: {failure_prob:.4f} ({failure_prob*100:.2f}%)

Video Metadata:
- Title: "{raw_features.get('title', 'N/A')}"
- Description: "{raw_features.get('description', 'N/A') or '(empty)'}"
- Tags: {raw_features.get('tags', [])}
- Category: {raw_features.get('category', 'N/A')}
- Duration: {raw_features.get('duration', 0)} seconds ({raw_features.get('duration', 0)/60:.1f} minutes)
- Upload Hour: {raw_features.get('upload_hour', 'N/A')}
- Channel Subscribers: {raw_features.get('channel_subscribers', 0):,}
- Channel Total Videos: {raw_features.get('channel_total_videos', 0):,}
- Channel Views: {raw_features.get('channel_views', 0):,}

Extracted Feature Details:
- Title Length: {features_dict.get('title_length', 0)} characters
- Title Word Count: {features_dict.get('title_word_count', 0)} words
- Title Has Question Mark: {'Yes' if features_dict.get('title_has_question', 0) else 'No'}
- Title Has Exclamation: {'Yes' if features_dict.get('title_has_exclamation', 0) else 'No'}
- Title Uppercase Ratio: {features_dict.get('title_uppercase_ratio', 0):.2%}
- Description Length: {features_dict.get('description_length', 0)} characters
- Description Word Count: {features_dict.get('description_word_count', 0)} words
- Has Description: {'Yes' if features_dict.get('has_description', 0) else 'No'}
- Number of Tags: {features_dict.get('num_tags', 0)}
- Duration: {features_dict.get('duration_minutes', 0):.1f} minutes
- Is Short Video (<5 min): {'Yes' if features_dict.get('is_short_video', 0) else 'No'}
- Is Long Video (>20 min): {'Yes' if features_dict.get('is_long_video', 0) else 'No'}
- Upload Day of Week: {features_dict.get('day_0', 0) or features_dict.get('day_1', 0) or features_dict.get('day_2', 0) or features_dict.get('day_3', 0) or features_dict.get('day_4', 0) or features_dict.get('day_5', 0) or features_dict.get('day_6', 0)}

Please analyze this prediction and provide actionable feedback."""

        # Build messages
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history if provided
        if conversation_history:
            for msg in conversation_history[-3:]:  # Last 3 messages for context
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role in ["user", "assistant"] and content:
                    messages.append({"role": role, "content": content})
        
        # Add current analysis request
        messages.append({"role": "user", "content": user_message})
        
        # Call LLM
        logger.info("Calling LLM for prediction analysis...")
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        
        analysis = response.choices[0].message.content
        if not analysis:
            analysis = "Unable to generate analysis at this time."
        
        logger.info(f"Generated analysis: {analysis[:100]}...")
        return analysis.strip()
        
    except Exception as e:
        logger.error(f"Failed to generate analysis: {str(e)}", exc_info=True)
        # Fallback analysis
        success_prob = prediction_output.get("success_probability", 0.0)
        failure_prob = prediction_output.get("failure_probability", prediction_output.get("not_success_probability", 1.0 - success_prob))
        if success_prob >= 0.8:
            return f"Great news! The model predicts strong success potential ({success_prob*100:.1f}% chance of success, {failure_prob*100:.1f}% chance of failure). Your video shows promising characteristics."
        elif success_prob < 0.5:
            return f"We feel this video may not be successful with a {failure_prob*100:.1f}% chance of failure ({success_prob*100:.1f}% chance of success). Consider improving your title, tags, and description to increase engagement potential."
        else:
            return f"The model predicts moderate success potential ({success_prob*100:.1f}% chance of success, {failure_prob*100:.1f}% chance of failure). There's room for improvement in your video metadata to boost performance."

