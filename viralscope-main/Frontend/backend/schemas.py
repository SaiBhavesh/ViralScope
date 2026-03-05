"""
Pydantic schemas for request/response validation.
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class VideoFeatures(BaseModel):
    """Input schema for video prediction features."""
    title: str = Field(..., description="Video title")
    description: Optional[str] = Field(default="", description="Video description")
    tags: List[str] = Field(default_factory=list, description="List of video tags")
    category: str = Field(default="Entertainment", description="Video category")
    duration: int = Field(..., ge=1, description="Video duration in seconds")
    upload_hour: int = Field(..., ge=0, le=23, description="Hour of upload (0-23)")
    # Optional channel features (defaults to 0 if not provided)
    channel_subscribers: Optional[int] = Field(default=0, ge=0, description="Channel subscriber count")
    channel_total_videos: Optional[int] = Field(default=0, ge=0, description="Channel total videos")
    channel_views: Optional[int] = Field(default=0, ge=0, description="Channel total views")
    upload_day_of_week: Optional[int] = Field(default=None, ge=0, le=6, description="Day of week (0=Monday, 6=Sunday)")


class PredictionResponse(BaseModel):
    """Response schema for prediction results."""
    success_probability: float = Field(..., ge=0.0, le=1.0, description="Probability of success (0-1)")
    failure_probability: float = Field(..., ge=0.0, le=1.0, description="Probability of failure (0-1)")
    predicted_class: int = Field(..., ge=0, le=1, description="Predicted class (0=failure, 1=success)")
    analysis: Optional[str] = Field(default=None, description="LLM-generated analysis and feedback on the prediction")


class ChatMessage(BaseModel):
    """Chat message in conversation history."""
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class AgentRequest(BaseModel):
    """Request schema for LLM agent endpoint."""
    user_query: str = Field(..., description="Natural language query describing the video")
    conversation_history: Optional[List[ChatMessage]] = Field(default=None, description="Previous conversation messages for context")


class ConversationalResponse(BaseModel):
    """Response schema for conversational LLM responses."""
    conversational: bool = Field(default=True, description="Whether this is a conversational response")
    response: str = Field(..., description="LLM conversational response")
    requires_prediction: bool = Field(default=False, description="Whether prediction was requested")


class AgentResponse(BaseModel):
    """Response schema for LLM agent endpoint."""
    # For prediction responses
    parsed_input: Optional[VideoFeatures] = Field(default=None, description="Extracted structured video features")
    success_probability: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Probability of success (0-1)")
    failure_probability: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Probability of failure (0-1)")
    predicted_class: Optional[int] = Field(default=None, ge=0, le=1, description="Predicted class (0=failure, 1=success)")
    analysis: Optional[str] = Field(default=None, description="LLM-generated analysis and feedback on the prediction")
    # For conversational responses
    conversational: Optional[bool] = Field(default=None, description="Whether this is a conversational response")
    response: Optional[str] = Field(default=None, description="LLM conversational response")
    requires_prediction: bool = Field(..., description="Whether prediction was requested")



