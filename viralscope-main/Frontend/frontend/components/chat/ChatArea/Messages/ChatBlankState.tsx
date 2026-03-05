'use client'

import { motion } from 'framer-motion'

const ChatBlankState = () => {
  return (
    <section
      className="flex flex-col items-center text-center font-geist"
      aria-label="Welcome message"
    >
      <div className="flex max-w-3xl flex-col gap-y-8">
        <motion.h1
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.3 }}
          className="text-3xl font-[600] tracking-tight text-primary"
        >
          <div className="flex items-center justify-center gap-x-2 whitespace-nowrap font-medium">
            <span className="flex items-center font-[600]">
              Welcome to ViralScope
            </span>
          </div>
          <p className="mt-4 text-lg text-muted">
            Ask questions about YouTube videos or get predictions for your video's view potential
          </p>
        </motion.h1>
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.5 }}
          className="flex flex-col gap-2 text-sm text-muted"
        >
          <p>Try asking:</p>
          <ul className="list-disc list-inside space-y-1">
            <li>"What makes a YouTube video go viral?"</li>
            <li>"Predict views for my YouTube video"</li>
            <li>"How can I improve my YouTube video's performance?"</li>
          </ul>
        </motion.div>
      </div>
    </section>
  )
}

export default ChatBlankState

