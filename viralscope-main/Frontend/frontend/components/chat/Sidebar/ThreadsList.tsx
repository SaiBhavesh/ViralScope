'use client'

import { useState, useCallback } from 'react'
import { useRouter } from 'next/navigation'
import { useStore } from '@/store'
import ThreadItem from './ThreadItem'
import { Skeleton } from '@/components/ui/skeleton'
import { cn } from '@/lib/utils'

const ThreadsList = () => {
  const router = useRouter()
  const { threads, currentThreadId, setCurrentThreadId } = useStore()
  const [editingId, setEditingId] = useState<string | null>(null)

  const handleThreadClick = useCallback(
    (id: string) => {
      setCurrentThreadId(id)
      router.push(`/chat/${id}`)
    },
    [setCurrentThreadId, router]
  )

  if (threads.length === 0) {
    return (
      <div className="w-full">
        <div className="mb-2 text-xs font-medium uppercase text-muted">Chats</div>
        <div className="mt-4 text-sm text-muted">
          <p>No chats yet. Start a new conversation!</p>
        </div>
      </div>
    )
  }

  return (
    <div className="w-full">
      <div className="mb-2 text-xs font-medium uppercase text-muted">Chats</div>
      <div className="h-[calc(100vh-200px)] overflow-y-auto space-y-1">
        {threads.map((thread) => (
          <ThreadItem
            key={thread.id}
            thread={thread}
            isSelected={currentThreadId === thread.id}
            isEditing={editingId === thread.id}
            onSelect={() => handleThreadClick(thread.id)}
            onEditStart={() => setEditingId(thread.id)}
            onEditEnd={() => setEditingId(null)}
          />
        ))}
      </div>
    </div>
  )
}

export default ThreadsList

