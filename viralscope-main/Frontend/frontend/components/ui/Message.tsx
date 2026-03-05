'use client';

import { cn } from '@/lib/utils';
import { User, Bot } from 'lucide-react';
import type { AgentResponse } from '@/lib/types';
import { useEffect, useRef } from 'react';

interface MessageProps {
  role: 'user' | 'assistant';
  content: string;
  data?: AgentResponse;
  isLoading?: boolean;
}

export function Message({ role, content, data, isLoading }: MessageProps) {
  const isUser = role === 'user';
  const progressBarRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Trigger animation when component mounts
    if (progressBarRef.current && data?.success_probability !== undefined) {
      const successBar = progressBarRef.current.querySelector('[data-success-bar]') as HTMLElement;
      const failureBar = progressBarRef.current.querySelector('[data-failure-bar]') as HTMLElement;
      
      if (successBar && failureBar) {
        // Calculate failure probability from success (failure = 1 - success)
        const successProb = data.success_probability;
        const failureProb = 1 - successProb;
        
        // Reset to 0 for animation
        successBar.style.width = '0%';
        failureBar.style.width = '0%';
        
        // Animate to target width
        setTimeout(() => {
          successBar.style.width = `${successProb * 100}%`;
          failureBar.style.width = `${failureProb * 100}%`;
        }, 50);
      }
    }
  }, [data?.success_probability]);

  return (
    <div className={cn(
      "flex gap-4 p-4",
      isUser ? "justify-end" : "justify-start"
    )}>
      {!isUser && (
        <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
          <Bot className="w-5 h-5 text-white" />
        </div>
      )}
      
      <div className={cn(
        "flex flex-col gap-2 max-w-[80%]",
        isUser ? "items-end" : "items-start"
      )}>
        <div className={cn(
          "rounded-2xl px-4 py-3 shadow-md",
          isUser 
            ? "bg-gradient-to-br from-blue-600 to-blue-700 text-white" 
            : "bg-gradient-to-br from-gray-50 to-gray-100 text-gray-900 dark:from-gray-800 dark:to-gray-900 dark:text-gray-100"
        )}>
          {isLoading ? (
            <div className="flex items-center gap-2">
              <div className="animate-spin rounded-full h-4 w-4 border-2 border-gray-400 border-t-transparent"></div>
              <span className="text-sm">Thinking...</span>
            </div>
          ) : (
            <div className="prose prose-sm dark:prose-invert max-w-none">
              <p className="text-sm whitespace-pre-wrap break-words leading-relaxed">{content}</p>
            </div>
          )}
        </div>

        {/* Prediction Results */}
        {!isUser && data?.requires_prediction && data.success_probability !== undefined && data.parsed_input && (
          <div className="w-full mt-2 space-y-3">
            <div className={`bg-gradient-to-br rounded-xl p-5 border-2 shadow-lg ${
              data.predicted_class === 1 
                ? 'from-green-50 to-emerald-50 dark:from-green-900/30 dark:to-emerald-900/30 border-green-300 dark:border-green-700'
                : 'from-orange-50 to-red-50 dark:from-orange-900/30 dark:to-red-900/30 border-orange-300 dark:border-orange-700'
            }`}>
              <div className="flex items-center justify-between mb-4">
                <h3 className={`text-base font-bold ${
                  data.predicted_class === 1
                    ? 'text-green-800 dark:text-green-200'
                    : 'text-orange-800 dark:text-orange-200'
                }`}>
                  📊 Prediction Results
                </h3>
                <div className={`px-3 py-1 rounded-full text-xs font-semibold ${
                  data.predicted_class === 1
                    ? 'bg-green-200 dark:bg-green-800 text-green-800 dark:text-green-200'
                    : 'bg-orange-200 dark:bg-orange-800 text-orange-800 dark:text-orange-200'
                }`}>
                  {data.predicted_class === 1 ? '✅ SUCCESS' : '❌ FAILURE'}
                </div>
              </div>

              {/* Animated Progress Bar */}
              <div className="mb-4" ref={progressBarRef}>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs font-semibold text-gray-700 dark:text-gray-300 flex items-center gap-1.5">
                    <span className="w-2.5 h-2.5 rounded-full bg-green-500 shadow-sm"></span>
                    Success Rate
                  </span>
                  <span className="text-xs font-semibold text-gray-700 dark:text-gray-300">
                    {(data.success_probability * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="relative h-12 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden shadow-inner border-2 border-gray-300 dark:border-gray-600">
                  {/* Success portion (green) - animates from left */}
                  <div 
                    data-success-bar
                    className="absolute left-0 top-0 h-full bg-gradient-to-r from-green-500 via-green-400 to-emerald-500 rounded-full transition-all duration-1000 ease-out flex items-center justify-end pr-4 shadow-lg"
                    style={{ 
                      width: `${data.success_probability * 100}%`,
                      transition: 'width 1.2s cubic-bezier(0.4, 0, 0.2, 1)'
                    }}
                  >
                    {data.success_probability > 0.1 && (
                      <span className="text-sm font-bold text-white drop-shadow-lg">
                        {(data.success_probability * 100).toFixed(1)}%
                      </span>
                    )}
                  </div>
                  {/* Remaining portion (neutral gray) - shows what's left to achieve */}
                  <div 
                    data-failure-bar
                    className="absolute right-0 top-0 h-full bg-gradient-to-l from-gray-300 to-gray-400 dark:from-gray-600 dark:to-gray-500 rounded-full transition-all duration-1000 ease-out"
                    style={{ 
                      width: `${(1 - data.success_probability) * 100}%`,
                      transition: 'width 1.2s cubic-bezier(0.4, 0, 0.2, 1)'
                    }}
                  />
                </div>
              </div>

              {/* Success Stat Card */}
              <div className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm rounded-lg p-4 border border-gray-200 dark:border-gray-700">
                <div className="text-xs text-gray-600 dark:text-gray-400 mb-2 uppercase tracking-wide">Success Probability</div>
                <div className={`text-4xl font-bold ${
                  data.predicted_class === 1
                    ? 'text-green-700 dark:text-green-400'
                    : data.success_probability >= 0.5
                      ? 'text-green-600 dark:text-green-500'
                      : 'text-gray-700 dark:text-gray-300'
                }`}>
                  {(data.success_probability * 100).toFixed(1)}%
                </div>
                {data.predicted_class === 1 && (
                  <div className="mt-2 text-xs text-green-600 dark:text-green-400 font-medium">
                    🎉 Great potential for success!
                  </div>
                )}
              </div>
            </div>

            {/* Video Features */}
            <div className="bg-white/90 dark:bg-gray-800/90 backdrop-blur-sm rounded-xl p-4 border border-gray-200 dark:border-gray-700 shadow-sm">
              <h4 className="text-xs font-semibold text-gray-700 dark:text-gray-300 mb-3 flex items-center gap-2">
                <span>📹</span>
                <span>Video Details</span>
              </h4>
              <div className="space-y-3 text-xs">
                <div className="pb-2 border-b border-gray-200 dark:border-gray-700">
                  <span className="text-gray-500 dark:text-gray-400 text-[10px] uppercase tracking-wide">Title</span>
                  <p className="text-gray-900 dark:text-gray-100 font-medium mt-1">
                    {data.parsed_input.title}
                  </p>
                </div>
                {data.parsed_input.description && (
                  <div className="pb-2 border-b border-gray-200 dark:border-gray-700">
                    <span className="text-gray-500 dark:text-gray-400 text-[10px] uppercase tracking-wide">Description</span>
                    <p className="text-gray-900 dark:text-gray-100 mt-1 line-clamp-2">
                      {data.parsed_input.description}
                    </p>
                  </div>
                )}
                <div className="flex items-center gap-2">
                  <span className="text-gray-500 dark:text-gray-400 text-[10px] uppercase tracking-wide">Category</span>
                  <span className="px-2.5 py-1 bg-gradient-to-r from-blue-100 to-blue-200 dark:from-blue-900 dark:to-blue-800 text-blue-800 dark:text-blue-200 rounded-md text-xs font-semibold shadow-sm">
                    {data.parsed_input.category}
                  </span>
                </div>
                {data.parsed_input.tags.length > 0 && (
                  <div className="flex flex-wrap gap-1.5 items-center">
                    <span className="text-gray-500 dark:text-gray-400 text-[10px] uppercase tracking-wide">Tags</span>
                    {data.parsed_input.tags.map((tag, idx) => (
                      <span
                        key={idx}
                        className="px-2 py-0.5 bg-gradient-to-r from-gray-100 to-gray-200 dark:from-gray-700 dark:to-gray-600 text-gray-700 dark:text-gray-300 rounded-md text-xs font-medium shadow-sm"
                      >
                        {tag}
                      </span>
                    ))}
                  </div>
                )}
                <div className="flex gap-4 pt-2 border-t border-gray-200 dark:border-gray-700">
                  <div className="flex items-center gap-1.5">
                    <span className="text-gray-500 dark:text-gray-400">⏱️</span>
                    <span className="text-gray-700 dark:text-gray-300 font-medium">
                      {Math.floor(data.parsed_input.duration / 60)}m {data.parsed_input.duration % 60}s
                    </span>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <span className="text-gray-500 dark:text-gray-400">🕐</span>
                    <span className="text-gray-700 dark:text-gray-300 font-medium">
                      {data.parsed_input.upload_hour}:00
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {isUser && (
        <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gradient-to-br from-gray-400 to-gray-600 flex items-center justify-center">
          <User className="w-5 h-5 text-white" />
        </div>
      )}
    </div>
  );
}



