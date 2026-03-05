/**
 * Client for communicating with the LLM agent API.
 */
import type { AgentRequest, AgentResponse, VideoFeatures, PredictionResponse } from './types';

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';

export class ApiError extends Error {
  constructor(
    message: string,
    public statusCode?: number
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

/**
 * Send natural language query to the agent endpoint and get predictions.
 * 
 * @param userQuery - Natural language description of the video
 * @param conversationHistory - Optional conversation history for context
 * @returns Promise resolving to AgentResponse with parsed input and predictions
 * @throws ApiError if the API call fails
 */
export async function predictVideo(userQuery: string, conversationHistory?: Array<{role: 'user' | 'assistant', content: string}>): Promise<AgentResponse> {
  if (!userQuery.trim()) {
    throw new ApiError('Query cannot be empty');
  }

  const request: AgentRequest = {
    user_query: userQuery,
    conversation_history: conversationHistory,
  };

  try {
    const response = await fetch(`${BACKEND_URL}/agent`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
      throw new ApiError(
        errorData.detail || `API request failed with status ${response.status}`,
        response.status
      );
    }

    const data: AgentResponse = await response.json();
    return data;
  } catch (error) {
    if (error instanceof ApiError) {
      throw error;
    }
    if (error instanceof Error) {
      throw new ApiError(`Network error: ${error.message}`);
    }
    throw new ApiError('Unknown error occurred');
  }
}

/**
 * Direct prediction endpoint accepting structured video features.
 * 
 * @param features - Structured video features
 * @returns Promise resolving to PredictionResponse with predicted views and confidence
 * @throws ApiError if the API call fails
 */
export async function predictVideoDirect(features: VideoFeatures): Promise<PredictionResponse> {
  try {
    const response = await fetch(`${BACKEND_URL}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(features),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
      throw new ApiError(
        errorData.detail || `API request failed with status ${response.status}`,
        response.status
      );
    }

    const data: PredictionResponse = await response.json();
    return data;
  } catch (error) {
    if (error instanceof ApiError) {
      throw error;
    }
    if (error instanceof Error) {
      throw new ApiError(`Network error: ${error.message}`);
    }
    throw new ApiError('Unknown error occurred');
  }
}



