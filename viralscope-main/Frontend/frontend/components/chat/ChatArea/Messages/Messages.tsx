import type { ChatMessage } from '@/store'
import { AgentMessage, UserMessage } from './MessageItem'
import ChatBlankState from './ChatBlankState'
import PredictionForm from './PredictionForm'
import Icon from '@/components/ui/icon'
import { memo } from 'react'

interface MessageListProps {
  messages: ChatMessage[]
}

const Messages = ({ messages }: MessageListProps) => {
  if (messages.length === 0) {
    return <ChatBlankState />
  }

  return (
    <>
      {messages.map((message, index) => {
        const key = `${message.role}-${message.id}-${index}`
        const isLastMessage = index === messages.length - 1

        if (message.role === 'form' && message.showForm) {
          return (
            <div key={key} className="flex flex-row items-start gap-4 font-geist">
              <div className="flex-shrink-0">
                <Icon type="agent" size="sm" />
              </div>
              <div className="flex-1">
                <PredictionForm messageId={message.id} />
              </div>
            </div>
          )
        }

        if (message.role === 'assistant') {
          return (
            <AgentMessage
              key={key}
              message={message}
              isLastMessage={isLastMessage}
            />
          )
        }
        return <UserMessage key={key} message={message} />
      })}
    </>
  )
}

export default Messages

