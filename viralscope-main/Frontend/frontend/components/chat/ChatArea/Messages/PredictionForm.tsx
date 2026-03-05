'use client'

import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { TextArea } from '@/components/ui/textarea'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { useStore } from '@/store'
import Icon from '@/components/ui/icon'
import { predictVideoDirect } from '@/lib/llmClient'
import { toast } from 'sonner'
import type { VideoFeatures } from '@/lib/types'

interface PredictionFormProps {
  messageId: string
  onClose?: () => void
}

const CATEGORIES = [
  'Entertainment',
  'Education',
  'Gaming',
  'Music',
  'Sports',
  'Technology',
  'Lifestyle',
  'Comedy',
  'News',
  'Travel',
  'Food',
  'Fashion',
  'Beauty',
  'Science',
  'Other'
]

const PredictionForm = ({ messageId, onClose }: PredictionFormProps) => {
  const { setMessages, setIsStreaming, isStreaming } = useStore()
  const [formData, setFormData] = useState<Partial<VideoFeatures>>({
    title: '',
    description: '',
    tags: [],
    category: 'Entertainment',
    duration: undefined,
    upload_hour: 12
  })
  const [tagInput, setTagInput] = useState('')
  const [errors, setErrors] = useState<Record<string, string>>({})

  const validateForm = (): boolean => {
    const newErrors: Record<string, string> = {}

    if (!formData.title || formData.title.trim() === '') {
      newErrors.title = 'Title is required'
    }

    if (!formData.duration || formData.duration <= 0) {
      newErrors.duration = 'Duration must be greater than 0'
    }

    if (formData.upload_hour === undefined || formData.upload_hour < 0 || formData.upload_hour > 23) {
      newErrors.upload_hour = 'Upload hour must be between 0 and 23'
    }

    setErrors(newErrors)
    return Object.keys(newErrors).length === 0
  }

  const handleAddTag = () => {
    if (tagInput.trim() && !formData.tags?.includes(tagInput.trim())) {
      setFormData({
        ...formData,
        tags: [...(formData.tags || []), tagInput.trim()]
      })
      setTagInput('')
    }
  }

  const handleRemoveTag = (tagToRemove: string) => {
    setFormData({
      ...formData,
      tags: formData.tags?.filter(tag => tag !== tagToRemove) || []
    })
  }

  const handleSubmit = async () => {
    if (!validateForm()) {
      toast.error('Please fill in all required fields')
      return
    }

    setIsStreaming(true)

    try {
      // Convert form data to the format expected by the API
      const videoFeatures: VideoFeatures = {
        title: formData.title!.trim(),
        description: formData.description || '',
        tags: formData.tags || [],
        category: formData.category || 'Entertainment',
        duration: formData.duration!,
        upload_hour: formData.upload_hour || 12
      }

      // Create a user message showing the submitted form data
      const userMessage = {
        id: Date.now().toString(),
        role: 'user' as const,
        content: `I want to predict views for my YouTube video:\n\n**Title:** ${videoFeatures.title}\n**Duration:** ${videoFeatures.duration} seconds\n**Category:** ${videoFeatures.category}\n**Upload Hour:** ${videoFeatures.upload_hour}:00${videoFeatures.description ? `\n**Description:** ${videoFeatures.description}` : ''}${videoFeatures.tags.length > 0 ? `\n**Tags:** ${videoFeatures.tags.join(', ')}` : ''}`,
        data: { formData: videoFeatures },
        timestamp: new Date()
      }

      // Call direct prediction endpoint
      const result = await predictVideoDirect(videoFeatures)

      // Build assistant message content with prediction and analysis (only success, no failure)
      const successPercent = (result.success_probability * 100).toFixed(1)
      const predictionText = result.predicted_class === 1 
        ? `✅ **Predicted: SUCCESS** (${successPercent}% success rate)`
        : `📊 **Predicted: Needs Improvement** (${successPercent}% success rate)`
      
      let assistantContent = `I've analyzed your YouTube video and generated a prediction!\n\n${predictionText}\n\n**Success Rate:** ${successPercent}%`
      
      // Add LLM analysis if available
      if (result.analysis) {
        assistantContent += `\n\n---\n\n${result.analysis}`
      }

      // Replace the form message with user message and add the prediction result
      setMessages((prev) => {
        const filtered = prev.filter(msg => msg.id !== messageId)
        const assistantMessage = {
          id: (Date.now() + 1).toString(),
          role: 'assistant' as const,
          content: assistantContent,
          data: { ...result, parsed_input: videoFeatures },
          timestamp: new Date()
        }
        return [...filtered, userMessage, assistantMessage]
      })

      if (onClose) {
        onClose()
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'An unexpected error occurred'
      toast.error(`Error: ${errorMessage}`)
    } finally {
      setIsStreaming(false)
    }
  }

  return (
    <div className="flex flex-col gap-4 rounded-xl border border-border bg-background-secondary p-6 font-geist">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-primary">YouTube Video Prediction</h3>
        {onClose && (
          <Button
            variant="ghost"
            size="icon"
            onClick={onClose}
            className="h-6 w-6"
          >
            <Icon type="x" size="xs" />
          </Button>
        )}
      </div>

      <div className="space-y-4">
        {/* Title */}
        <div className="space-y-2">
          <label className="text-sm font-medium text-primary">
            YouTube Video Title <span className="text-destructive">*</span>
          </label>
          <input
            type="text"
            value={formData.title || ''}
            onChange={(e) => setFormData({ ...formData, title: e.target.value })}
            className="w-full rounded-xl border border-border bg-primaryAccent px-4 py-2 text-sm text-primary placeholder:text-muted focus:border-primary/50 focus:outline-none"
            placeholder="Enter your YouTube video title"
          />
          {errors.title && <p className="text-xs text-destructive">{errors.title}</p>}
        </div>

        {/* Description */}
        <div className="space-y-2">
          <label className="text-sm font-medium text-primary">YouTube Description</label>
          <TextArea
            value={formData.description || ''}
            onChange={(e) => setFormData({ ...formData, description: e.target.value })}
            placeholder="Enter your YouTube video description (optional)"
            className="w-full border border-border bg-primaryAccent text-primary placeholder:text-muted"
          />
        </div>

        {/* Duration and Upload Hour */}
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <label className="text-sm font-medium text-primary">
              Duration (seconds) <span className="text-destructive">*</span>
            </label>
            <input
              type="number"
              value={formData.duration || ''}
              onChange={(e) => setFormData({ ...formData, duration: parseInt(e.target.value) || undefined })}
              className="w-full rounded-xl border border-border bg-primaryAccent px-4 py-2 text-sm text-primary placeholder:text-muted focus:border-primary/50 focus:outline-none"
              placeholder="e.g., 120"
              min="1"
            />
            {errors.duration && <p className="text-xs text-destructive">{errors.duration}</p>}
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium text-primary">
              Upload Hour (0-23) <span className="text-destructive">*</span>
            </label>
            <input
              type="number"
              value={formData.upload_hour ?? ''}
              onChange={(e) => setFormData({ ...formData, upload_hour: parseInt(e.target.value) || undefined })}
              className="w-full rounded-xl border border-border bg-primaryAccent px-4 py-2 text-sm text-primary placeholder:text-muted focus:border-primary/50 focus:outline-none"
              placeholder="e.g., 14"
              min="0"
              max="23"
            />
            {errors.upload_hour && <p className="text-xs text-destructive">{errors.upload_hour}</p>}
          </div>
        </div>

        {/* Category */}
        <div className="space-y-2">
          <label className="text-sm font-medium text-primary">YouTube Category</label>
          <Select
            value={formData.category || 'Entertainment'}
            onValueChange={(value) => setFormData({ ...formData, category: value })}
          >
            <SelectTrigger className="w-full border border-border bg-primaryAccent text-primary">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {CATEGORIES.map((cat) => (
                <SelectItem key={cat} value={cat}>
                  {cat}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* Tags */}
        <div className="space-y-2">
          <label className="text-sm font-medium text-primary">YouTube Tags</label>
          <div className="flex gap-2">
            <input
              type="text"
              value={tagInput}
              onChange={(e) => setTagInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  e.preventDefault()
                  handleAddTag()
                }
              }}
              className="flex-1 rounded-xl border border-border bg-primaryAccent px-4 py-2 text-sm text-primary placeholder:text-muted focus:border-primary/50 focus:outline-none"
              placeholder="Add YouTube tags (press Enter)"
            />
            <Button
              type="button"
              onClick={handleAddTag}
              variant="outline"
              className="rounded-xl"
            >
              <Icon type="plus-icon" size="xs" />
            </Button>
          </div>
          {formData.tags && formData.tags.length > 0 && (
            <div className="flex flex-wrap gap-2">
              {formData.tags.map((tag) => (
                <span
                  key={tag}
                  className="inline-flex items-center gap-1 rounded-full bg-accent px-3 py-1 text-xs text-primary"
                >
                  {tag}
                  <button
                    type="button"
                    onClick={() => handleRemoveTag(tag)}
                    className="hover:text-destructive"
                  >
                    <Icon type="x" size="xxs" />
                  </button>
                </span>
              ))}
            </div>
          )}
        </div>
      </div>

      <div className="flex justify-end gap-2 pt-2">
        {onClose && (
          <Button
            variant="ghost"
            onClick={onClose}
            disabled={isStreaming}
          >
            Cancel
          </Button>
        )}
        <Button
          onClick={handleSubmit}
          disabled={isStreaming}
          className="rounded-xl bg-primary text-primaryAccent hover:bg-primary/80"
        >
          {isStreaming ? 'Predicting...' : 'Predict YouTube Views'}
        </Button>
      </div>
    </div>
  )
}

export default PredictionForm

