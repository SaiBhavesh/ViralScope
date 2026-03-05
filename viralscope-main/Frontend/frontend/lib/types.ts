/**
 * TypeScript type definitions matching backend Pydantic schemas.
 */

export interface VideoFeatures {
  title: string;
  description: string;
  tags: string[];
  category: string;
  duration: number;
  upload_hour: number;
}

export interface PredictionResponse {
  success_probability: number;
  failure_probability: number;
  predicted_class: number; // 0 = failure, 1 = success
  analysis?: string; // LLM-generated analysis and feedback
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

export interface AgentRequest {
  user_query: string;
  conversation_history?: ChatMessage[];
}

export interface AgentResponse {
  // For prediction responses
  parsed_input?: VideoFeatures;
  success_probability?: number;
  failure_probability?: number;
  predicted_class?: number; // 0 = failure, 1 = success
  analysis?: string; // LLM-generated analysis and feedback
  // For conversational responses
  conversational?: boolean;
  response?: string;
  requires_prediction: boolean;
}

export type LoadingState = {
  status: 'loading';
};

export type ErrorState = {
  status: 'error';
  message: string;
};

export type SuccessState = {
  status: 'success';
  data: AgentResponse;
};

export type ComponentState = LoadingState | ErrorState | SuccessState | { status: 'idle' };



