'use client'

import { Suspense, useEffect } from 'react'
import { useParams, useRouter } from 'next/navigation'
import { ChatArea } from '@/components/chat/ChatArea'
import Sidebar from '@/components/chat/Sidebar/Sidebar'
import { useStore } from '@/store'
import { toast } from 'sonner'

export default function ChatPage() {
  const params = useParams()
  const router = useRouter()
  const { setHydrated, hydrated, threads, setCurrentThreadId, createThread } = useStore()
  const threadId = params?.threadId as string

  useEffect(() => {
    setHydrated()
  }, [setHydrated])

  useEffect(() => {
    if (hydrated && threadId) {
      const thread = threads.find((t) => t.id === threadId)
      if (thread) {
        setCurrentThreadId(threadId)
      } else if (threads.length === 0) {
        // No threads exist, create one
        const newThreadId = createThread()
        router.replace(`/chat/${newThreadId}`)
      } else {
        toast.error('Chat not found')
        router.push('/home')
      }
    }
  }, [hydrated, threadId, threads, setCurrentThreadId, createThread, router])

  if (!hydrated) {
    return <div>Loading...</div>
  }

  return (
    <Suspense fallback={<div>Loading...</div>}>
      <div className="flex h-screen bg-background/80">
        <Sidebar />
        <ChatArea />
      </div>
    </Suspense>
  )
}

