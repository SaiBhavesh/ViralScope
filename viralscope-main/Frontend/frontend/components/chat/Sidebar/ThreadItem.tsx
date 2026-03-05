'use client'

import { useState, useRef, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { Button } from '@/components/ui/button'
import Icon from '@/components/ui/icon'
import { useStore } from '@/store'
import { cn, truncateText } from '@/lib/utils'
import { toast } from 'sonner'
import type { ChatThread } from '@/store'

interface ThreadItemProps {
  thread: ChatThread
  isSelected: boolean
  isEditing: boolean
  onSelect: () => void
  onEditStart: () => void
  onEditEnd: () => void
}

const ThreadItem = ({
  thread,
  isSelected,
  isEditing,
  onSelect,
  onEditStart,
  onEditEnd
}: ThreadItemProps) => {
  const router = useRouter()
  const { deleteThread, updateThreadTitle } = useStore()
  const [title, setTitle] = useState(thread.title)
  const [isDeleting, setIsDeleting] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (isEditing && inputRef.current) {
      inputRef.current.focus()
      inputRef.current.select()
    }
  }, [isEditing])

  const handleSave = () => {
    if (title.trim()) {
      updateThreadTitle(thread.id, title.trim())
      onEditEnd()
    } else {
      setTitle(thread.title)
      onEditEnd()
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      handleSave()
    } else if (e.key === 'Escape') {
      setTitle(thread.title)
      onEditEnd()
    }
  }

  const handleDelete = async (e: React.MouseEvent) => {
    e.stopPropagation()
    if (isDeleting) return
    
    setIsDeleting(true)
    try {
      deleteThread(thread.id)
      toast.success('Chat deleted')
      // If this was the current thread, redirect to home
      if (isSelected) {
        router.push('/home')
      }
    } catch (error) {
      toast.error('Failed to delete chat')
    } finally {
      setIsDeleting(false)
    }
  }

  const handleDoubleClick = () => {
    if (!isEditing) {
      onEditStart()
    }
  }

  return (
    <div
      className={cn(
        'group flex h-11 w-full items-center justify-between rounded-lg px-3 py-2 transition-colors duration-200',
        isSelected
          ? 'cursor-default bg-primary/10'
          : 'cursor-pointer bg-background-secondary hover:bg-background-secondary/80'
      )}
      onClick={onSelect}
      onDoubleClick={handleDoubleClick}
    >
      {isEditing ? (
        <input
          ref={inputRef}
          type="text"
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          onBlur={handleSave}
          onKeyDown={handleKeyDown}
          className="flex-1 bg-transparent text-sm font-medium text-primary outline-none"
          onClick={(e) => e.stopPropagation()}
        />
      ) : (
        <div className="flex flex-1 flex-col gap-1">
          <h4
            className={cn(
              'text-sm font-medium',
              isSelected && 'text-primary'
            )}
          >
            {truncateText(thread.title, 25)}
          </h4>
        </div>
      )}
      <Button
        variant="ghost"
        size="icon"
        className="h-6 w-6 transform opacity-0 transition-all duration-200 ease-in-out group-hover:opacity-100"
        onClick={handleDelete}
        disabled={isDeleting}
      >
        <Icon type="trash" size="xs" />
      </Button>
    </div>
  )
}

export default ThreadItem

