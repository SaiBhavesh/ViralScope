'use client';

import { useEffect, useRef } from 'react';
import { Message } from './Message';
import type { AgentResponse } from '@/lib/types';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  data?: AgentResponse;
  timestamp: Date;
}

interface ChatContainerProps {
  messages: Message[];
  isLoading?: boolean;
}

export function ChatContainer({ messages, isLoading }: ChatContainerProps) {
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  return (
    <div className="flex-1 overflow-y-auto px-4 py-6">
      <div className="max-w-4xl mx-auto space-y-2">
        {messages.length === 0 && !isLoading && (
          <div className="flex flex-col items-center justify-center h-full text-center py-12">
            <div className="w-16 h-16 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center mb-4">
              <svg
                className="w-8 h-8 text-white"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M7 4v16M17 4v16M3 8h4m10 0h4M3 12h18M3 16h4m10 0h4M4 20h16a1 1 0 001-1V5a1 1 0 00-1-1H4a1 1 0 00-1 1v14a1 1 0 001 1z"
                />
              </svg>
            </div>
            <h2 className="text-2xl font-bold text-gray-900 dark:text-gray-100 mb-2">
              ViralScope
            </h2>
            <p className="text-gray-600 dark:text-gray-400 max-w-md">
              Ask questions about YouTube videos or request predictions. Try saying "predict views for my YouTube video"
            </p>
          </div>
        )}

        {messages.map((message) => (
          <Message
            key={message.id}
            role={message.role}
            content={message.content}
            data={message.data}
          />
        ))}

        {isLoading && (
          <Message
            role="assistant"
            content=""
            isLoading={true}
          />
        )}

        <div ref={messagesEndRef} />
      </div>
    </div>
  );
}



