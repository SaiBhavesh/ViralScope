'use client'

import { useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { useStore } from '@/store'

export default function RootPage() {
  const router = useRouter()
  const { setHydrated, threads } = useStore()

  useEffect(() => {
    setHydrated()
    
    // Redirect to home or latest thread
    if (threads.length > 0) {
      router.push(`/chat/${threads[0].id}`)
    } else {
      router.push('/home')
    }
  }, [setHydrated, router, threads])

  return <div>Loading...</div>
}
