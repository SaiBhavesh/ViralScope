'use client'
import { useState } from 'react'
import { toast } from 'sonner'
import { TextArea } from '@/components/ui/textarea'
import { Button } from '@/components/ui/button'
import { useStore } from '@/store'
import Icon from '@/components/ui/icon'
import { predictVideo } from '@/lib/llmClient'

const ChatInput = () => {
  const { chatInputRef, setIsStreaming, setMessages, isStreaming, getMessages } = useStore()
  const [inputMessage, setInputMessage] = useState('')

  // Check if user is requesting a prediction
  const isPredictionRequest = (query: string): boolean => {
    const predictionKeywords = [
      'predict', 'prediction', 'review', 'analyze', 'estimate',
      'how many views', 'viral potential', 'performance',
      'will it go viral', 'should I upload', 'what are the chances',
      'predict my video', 'want to predict', 'would like to predict',
      'i want to predict', 'can you predict', 'help me predict'
    ]
    const queryLower = query.toLowerCase()
    return predictionKeywords.some(keyword => queryLower.includes(keyword))
  }

  // Check if query has enough information for prediction
  const hasEnoughInfo = (query: string): boolean => {
    // Check for title mentions (more flexible)
    const hasTitle = /(title|name|called|titled|about|video is|video about|it's|it is).{0,30}(video|clip|content)/i.test(query) ||
                    /(my|the|this).{0,20}(video|clip)/i.test(query)
    
    // Check for duration mentions (numbers with time units or duration keywords)
    const hasDuration = /(\d+\s*(minute|min|second|sec|hour|hr|h|m|s|minutes|seconds|hours)|duration|length|long|runtime)/i.test(query)
    
    // Check for upload hour mentions
    const hasUploadHour = /(upload|post|publish).{0,30}(\d+|hour|time|at \d+|morning|afternoon|evening|night)/i.test(query)
    
    // Need at least title and duration for a valid prediction
    // If user explicitly mentions they want to predict but lacks info, show form
    return hasTitle && hasDuration
  }

  const handleSubmit = async () => {
    if (!inputMessage.trim() || isStreaming) return

    const currentMessage = inputMessage
    setInputMessage('')
    setIsStreaming(true)

    // Add user message
    const userMessage = {
      id: Date.now().toString(),
      role: 'user' as const,
      content: currentMessage,
      timestamp: new Date()
    }
    setMessages((prev) => [...prev, userMessage])

    // Check if user wants prediction but doesn't have enough info
    if (isPredictionRequest(currentMessage) && !hasEnoughInfo(currentMessage)) {
      // Show form immediately
      const formMessage = {
        id: (Date.now() + 1).toString(),
        role: 'form' as const,
        content: 'Please fill out the form below to get a prediction for your YouTube video.',
        timestamp: new Date(),
        showForm: true
      }
      setMessages((prev) => [...prev, formMessage])
      setIsStreaming(false)
      return
    }

    try {
      // Build conversation history from previous messages (excluding forms)
      // Include prediction data in assistant messages for context
      const previousMessages = getMessages()
        .filter(msg => msg.role === 'user' || msg.role === 'assistant')
        .slice(-5) // Last 5 messages for context
        .map(msg => {
          let content = msg.content
          
          // If assistant message has prediction data, include it in the content
          if (msg.role === 'assistant' && msg.data) {
            const data = msg.data
            if (data.success_probability !== undefined && data.parsed_input) {
              // Enhance content with full prediction context
              const successPercent = (data.success_probability * 100).toFixed(1)
              const prediction = data.predicted_class === 1 ? 'SUCCESS' : 'FAILURE'
              content = `${msg.content}\n\n[Previous Prediction Context: Video "${data.parsed_input.title}" predicted ${prediction} (${successPercent}% success probability). Details: ${data.parsed_input.duration}s, ${data.parsed_input.category}, upload at ${data.parsed_input.upload_hour}:00]`
            }
          }
          
          return {
            role: msg.role === 'user' ? 'user' as const : 'assistant' as const,
            content: content
          }
        })
      
      const result = await predictVideo(currentMessage, previousMessages)
      
      // Determine assistant response content
      let assistantContent = ''
      
      // Check if we have a valid prediction with all required fields
      const parsedInput = result.parsed_input
      
      // Check if data looks like defaults/incomplete (user didn't provide real info)
      const looksLikeDefaults = parsedInput && (
        parsedInput.title === 'Untitled Video' ||
        parsedInput.title.trim() === '' ||
        (parsedInput.duration === 300 && !currentMessage.match(/\d+\s*(minute|min|second|sec|hour|hr|h|m|s)/i)) ||
        (parsedInput.upload_hour === 12 && !currentMessage.match(/upload.*(\d+|hour|time|at \d+)/i))
      )
      
      const hasValidPrediction = 
        result.requires_prediction &&
        parsedInput !== undefined &&
        parsedInput.title &&
        parsedInput.title.trim() !== '' &&
        parsedInput.title !== 'Untitled Video' &&
        parsedInput.duration !== undefined &&
        parsedInput.duration > 0 &&
        parsedInput.upload_hour !== undefined &&
        parsedInput.upload_hour >= 0 &&
        parsedInput.upload_hour <= 23 &&
        result.success_probability !== undefined &&
        result.predicted_class !== undefined &&
        !looksLikeDefaults
      
      if (hasValidPrediction && parsedInput && result.success_probability !== undefined) {
        // Show prediction with details (only success, no failure)
        const successPercent = (result.success_probability * 100).toFixed(1)
        const predictionEmoji = result.predicted_class === 1 ? '✅' : '📊'
        const predictionText = result.predicted_class === 1 
          ? `**Predicted: SUCCESS** (${successPercent}% success rate)`
          : `**Predicted: Needs Improvement** (${successPercent}% success rate)`
        
        assistantContent = `I've analyzed your YouTube video and generated a prediction!\n\n${predictionEmoji} ${predictionText}\n\n**YouTube Video Details:**\n- Title: ${parsedInput.title}\n- Duration: ${parsedInput.duration} seconds\n- Category: ${parsedInput.category || 'Entertainment'}\n- Upload Hour: ${parsedInput.upload_hour}:00\n\n**Success Rate:** ${successPercent}%`
        
        // Add LLM analysis if available
        if (result.analysis) {
          assistantContent += `\n\n---\n\n${result.analysis}`
        }
      } else if (result.conversational && result.response) {
        // Show conversational response
        assistantContent = result.response
      } else if (result.requires_prediction && !hasValidPrediction) {
        // User requested prediction but we don't have all required data
        // Show a form message instead of text
        const formMessage = {
          id: (Date.now() + 1).toString(),
          role: 'form' as const,
          content: 'Please fill out the form below to get a prediction for your YouTube video.',
          timestamp: new Date(),
          showForm: true
        }
        setMessages((prev) => [...prev, formMessage])
        setIsStreaming(false)
        return
      } else {
        assistantContent = 'I received your request. Processing...'
      }

      const assistantMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant' as const,
        content: assistantContent,
        data: result,
        timestamp: new Date()
      }

      setMessages((prev) => [...prev, assistantMessage])
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'An unexpected error occurred'
      toast.error(`Error: ${errorMessage}`)
      
      const errorMsg = {
        id: (Date.now() + 1).toString(),
        role: 'assistant' as const,
        content: `Sorry, I encountered an error: ${errorMessage}`,
        timestamp: new Date()
      }
      setMessages((prev) => [...prev, errorMsg])
    } finally {
      setIsStreaming(false)
    }
  }

  return (
    <div className="relative mx-auto mb-1 flex w-full max-w-2xl items-end justify-center gap-x-2 font-geist">
      <TextArea
        placeholder={'Ask about YouTube videos or say "predict views for my YouTube video"...'}
        value={inputMessage}
        onChange={(e) => setInputMessage(e.target.value)}
        onKeyDown={(e) => {
          if (
            e.key === 'Enter' &&
            !e.nativeEvent.isComposing &&
            !e.shiftKey &&
            !isStreaming
          ) {
            e.preventDefault()
            handleSubmit()
          }
        }}
        className="w-full border border-accent bg-primaryAccent px-4 text-sm text-primary focus:border-accent"
        disabled={isStreaming}
        ref={chatInputRef as React.Ref<HTMLTextAreaElement>}
      />
      <Button
        onClick={handleSubmit}
        disabled={!inputMessage.trim() || isStreaming}
        size="icon"
        className="rounded-xl bg-primary p-5 text-primaryAccent"
      >
        <Icon type="send" color="primaryAccent" />
      </Button>
    </div>
  )
}

export default ChatInput

