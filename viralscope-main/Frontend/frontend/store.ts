import { create } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant' | 'form'
  content: string
  timestamp: Date
  data?: any
  showForm?: boolean
}

export interface ChatThread {
  id: string
  title: string
  messages: ChatMessage[]
  createdAt: Date
  updatedAt: Date
}

interface Store {
  hydrated: boolean
  setHydrated: () => void
  threads: ChatThread[]
  currentThreadId: string | null
  setCurrentThreadId: (id: string | null) => void
  createThread: (title?: string) => string
  deleteThread: (id: string) => void
  updateThreadTitle: (id: string, title: string) => void
  getCurrentThread: () => ChatThread | null
  getMessages: () => ChatMessage[]
  setMessages: (
    messages: ChatMessage[] | ((prevMessages: ChatMessage[]) => ChatMessage[])
  ) => void
  chatInputRef: React.RefObject<HTMLTextAreaElement>
  isStreaming: boolean
  setIsStreaming: (isStreaming: boolean) => void
}

export const useStore = create<Store>()(
  persist(
    (set, get) => ({
      hydrated: false,
      setHydrated: () => set({ hydrated: true }),
      threads: [],
      currentThreadId: null,
      setCurrentThreadId: (id) => set({ currentThreadId: id }),
      createThread: (title) => {
        const newThread: ChatThread = {
          id: Date.now().toString(),
          title: title || 'New Chat',
          messages: [],
          createdAt: new Date(),
          updatedAt: new Date()
        }
        set((state) => ({
          threads: [newThread, ...state.threads],
          currentThreadId: newThread.id
        }))
        return newThread.id
      },
      deleteThread: (id) => {
        set((state) => {
          const filtered = state.threads.filter((t) => t.id !== id)
          const newCurrentId =
            state.currentThreadId === id
              ? filtered.length > 0
                ? filtered[0].id
                : null
              : state.currentThreadId
          return {
            threads: filtered,
            currentThreadId: newCurrentId
          }
        })
      },
      updateThreadTitle: (id, title) => {
        set((state) => ({
          threads: state.threads.map((t) =>
            t.id === id ? { ...t, title, updatedAt: new Date() } : t
          )
        }))
      },
      getCurrentThread: () => {
        const state = get()
        if (!state.currentThreadId) return null
        return state.threads.find((t) => t.id === state.currentThreadId) || null
      },
      getMessages: () => {
        const state = get()
        const thread = state.getCurrentThread()
        return thread?.messages || []
      },
      setMessages: (messages) => {
        const state = get()
        const thread = state.getCurrentThread()
        if (!thread) return

        const newMessages =
          typeof messages === 'function' ? messages(thread.messages) : messages

        set((prevState) => ({
          threads: prevState.threads.map((t) =>
            t.id === thread.id
              ? {
                  ...t,
                  messages: newMessages,
                  updatedAt: new Date(),
                  title:
                    t.title === 'New Chat' && newMessages.length > 0
                      ? newMessages[0].content.slice(0, 50) || 'New Chat'
                      : t.title
                }
              : t
          )
        }))
      },
      chatInputRef: { current: null } as React.RefObject<HTMLTextAreaElement>,
      isStreaming: false,
      setIsStreaming: (isStreaming) => set(() => ({ isStreaming }))
    }),
    {
      name: 'chat-storage',
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        threads: state.threads,
        currentThreadId: state.currentThreadId
      }),
      onRehydrateStorage: () => (state) => {
        state?.setHydrated?.()
      }
    }
  )
)

