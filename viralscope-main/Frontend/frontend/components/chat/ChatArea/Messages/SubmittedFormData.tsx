'use client'

import Icon from '@/components/ui/icon'
import type { VideoFeatures } from '@/lib/types'
import { cn } from '@/lib/utils'

interface SubmittedFormDataProps {
  data: VideoFeatures
}

const SubmittedFormData = ({ data }: SubmittedFormDataProps) => {
  return (
    <div className="flex flex-col gap-3 rounded-xl border border-border bg-background-secondary p-4 font-geist">
      <div className="flex items-center gap-2">
        <Icon type="agent" size="xs" />
        <span className="text-sm font-semibold text-primary">Video Details Submitted</span>
      </div>
      
      <div className="grid grid-cols-1 gap-2 text-sm">
        <div className="flex items-start gap-2">
          <span className="font-medium text-muted">Title:</span>
          <span className="text-secondary">{data.title}</span>
        </div>
        
        {data.description && (
          <div className="flex items-start gap-2">
            <span className="font-medium text-muted">Description:</span>
            <span className="text-secondary">{data.description}</span>
          </div>
        )}
        
        <div className="grid grid-cols-2 gap-4">
          <div className="flex items-start gap-2">
            <span className="font-medium text-muted">Duration:</span>
            <span className="text-secondary">{data.duration} seconds</span>
          </div>
          
          <div className="flex items-start gap-2">
            <span className="font-medium text-muted">Upload Hour:</span>
            <span className="text-secondary">{data.upload_hour}:00</span>
          </div>
        </div>
        
        <div className="flex items-start gap-2">
          <span className="font-medium text-muted">Category:</span>
          <span className="text-secondary">{data.category}</span>
        </div>
        
        {data.tags && data.tags.length > 0 && (
          <div className="flex items-start gap-2">
            <span className="font-medium text-muted">Tags:</span>
            <div className="flex flex-wrap gap-1">
              {data.tags.map((tag) => (
                <span
                  key={tag}
                  className="inline-flex rounded-full bg-accent px-2 py-0.5 text-xs text-primary"
                >
                  {tag}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default SubmittedFormData

