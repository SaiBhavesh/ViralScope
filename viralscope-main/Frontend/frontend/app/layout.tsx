import type { Metadata } from 'next'
import { DM_Mono, Inter } from 'next/font/google'
import { Toaster } from '@/components/ui/sonner'
import './globals.css'

const inter = Inter({
  variable: '--font-geist-sans',
  weight: ['400', '500', '600'],
  subsets: ['latin']
})

const dmMono = DM_Mono({
  subsets: ['latin'],
  variable: '--font-dm-mono',
  weight: '400'
})

export const metadata: Metadata = {
  title: 'ViralScope - YouTube Video Predictor',
  description: 'Predict your YouTube video\'s viral potential with AI-powered analysis',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={`${inter.variable} ${dmMono.variable} antialiased`}>
        {children}
        <Toaster />
      </body>
    </html>
  )
}



