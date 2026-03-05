'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import { motion } from 'framer-motion'
import { Button } from '@/components/ui/button'
import Icon from '@/components/ui/icon'
import { useStore } from '@/store'
import ThreadsList from './ThreadsList'
import { cn } from '@/lib/utils'

const Sidebar = () => {
  const router = useRouter()
  const [isCollapsed, setIsCollapsed] = useState(false)
  const { createThread } = useStore()

  const handleNewChat = () => {
    const threadId = createThread()
    router.push(`/chat/${threadId}`)
  }

  return (
    <motion.aside
      className="relative flex h-screen shrink-0 grow-0 flex-col overflow-hidden border-r border-border px-2 py-3 font-dmmono"
      initial={{ width: '16rem' }}
      animate={{ width: isCollapsed ? '2.5rem' : '16rem' }}
      transition={{ type: 'spring', stiffness: 300, damping: 30 }}
    >
      <motion.button
        onClick={() => setIsCollapsed(!isCollapsed)}
        className="absolute right-2 top-2 z-10 p-1"
        aria-label={isCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        type="button"
        whileTap={{ scale: 0.95 }}
      >
        <Icon
          type="sheet"
          size="xs"
          className={cn('transform transition-transform', isCollapsed ? 'rotate-180' : 'rotate-0')}
        />
      </motion.button>

      <motion.div
        className="w-60 space-y-5"
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: isCollapsed ? 0 : 1, x: isCollapsed ? -20 : 0 }}
        transition={{ duration: 0.3, ease: 'easeInOut' }}
        style={{
          pointerEvents: isCollapsed ? 'none' : 'auto'
        }}
      >
        {/* Header */}
        <div className="flex items-center gap-2">
          <Icon type="agno" size="xs" />
          <span className="text-xs font-medium uppercase text-primary">ViralScope</span>
        </div>

        {/* New Chat Button */}
        <Button
          onClick={handleNewChat}
          size="lg"
          className="h-9 w-full rounded-xl bg-primary text-xs font-medium text-primaryAccent hover:bg-primary/80"
        >
          <Icon type="plus-icon" size="xs" className="text-primaryAccent" />
          <span className="uppercase">New Chat</span>
        </Button>

        {/* Threads List */}
        <ThreadsList />
      </motion.div>
    </motion.aside>
  )
}

export default Sidebar

