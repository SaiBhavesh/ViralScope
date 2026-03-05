'use client'

import { useEffect } from 'react'
import { motion } from 'framer-motion'
import { useRouter } from 'next/navigation'
import { Button } from '@/components/ui/button'
import Icon from '@/components/ui/icon'
import { useStore } from '@/store'

export default function HomePage() {
  const router = useRouter()
  const { createThread, threads, setHydrated } = useStore()

  useEffect(() => {
    setHydrated()
  }, [setHydrated])

  const handleNewChat = () => {
    const threadId = createThread()
    router.push(`/chat/${threadId}`)
  }

  return (
    <div className="flex h-screen flex-col items-center justify-center bg-background/80 p-8">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="flex max-w-2xl flex-col items-center gap-8 text-center"
      >
        <motion.div
          initial={{ scale: 0.9 }}
          animate={{ scale: 1 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="flex items-center gap-3"
        >
          <Icon type="agno" size="lg" />
          <h1 className="text-4xl font-bold text-primary">ViralScope</h1>
        </motion.div>

        <p className="text-lg text-muted">
          Predict your YouTube video's viral potential with AI-powered analysis
        </p>

        <div className="flex flex-col gap-4">
          <Button
            onClick={handleNewChat}
            size="lg"
            className="h-12 rounded-xl bg-primary px-8 text-base font-medium text-primaryAccent hover:bg-primary/80"
          >
            <Icon type="plus-icon" size="sm" className="text-primaryAccent" />
            Start New Chat
          </Button>

          {threads.length > 0 && (
            <div className="mt-8 w-full">
              <h2 className="mb-4 text-sm font-medium uppercase text-muted">Recent Chats</h2>
              <div className="space-y-2">
                {threads.slice(0, 5).map((thread) => (
                  <motion.div
                    key={thread.id}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.3 }}
                  >
                    <Button
                      variant="ghost"
                      onClick={() => router.push(`/chat/${thread.id}`)}
                      className="w-full justify-start rounded-lg bg-background-secondary px-4 py-3 text-left hover:bg-background-secondary/80"
                    >
                      <div className="flex flex-1 flex-col gap-1">
                        <span className="text-sm font-medium text-primary">
                          {thread.title}
                        </span>
                        <span className="text-xs text-muted">
                          {new Date(thread.updatedAt).toLocaleDateString()}
                        </span>
                      </div>
                    </Button>
                  </motion.div>
                ))}
              </div>
            </div>
          )}
        </div>

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
          className="mt-12 grid grid-cols-3 gap-6 text-sm text-muted"
        >
          <div className="flex flex-col gap-2">
            <Icon type="agent" size="md" />
            <p className="font-medium">AI Powered</p>
            <p className="text-xs">ML models trained on YouTube data</p>
          </div>
          <div className="flex flex-col gap-2">
            <Icon type="references" size="md" />
            <p className="font-medium">View Predictions</p>
            <p className="text-xs">Accurate YouTube view forecasts</p>
          </div>
          <div className="flex flex-col gap-2">
            <Icon type="reasoning" size="md" />
            <p className="font-medium">Growth Insights</p>
            <p className="text-xs">Actionable YouTube strategies</p>
          </div>
        </motion.div>
      </motion.div>
    </div>
  )
}

