import type { Metadata } from 'next'
import { DM_Sans, JetBrains_Mono } from 'next/font/google'
import { ClerkProvider } from '@clerk/nextjs'
import { clerkEnabled } from '@/lib/auth'
import { SITE } from '@/lib/site'
import './globals.css'

const dmSans = DM_Sans({
  subsets: ['latin'],
  variable: '--font-dm-sans',
  display: 'swap',
})

const jetbrainsMono = JetBrains_Mono({
  subsets: ['latin'],
  variable: '--font-jetbrains-mono',
  display: 'swap',
})

export const metadata: Metadata = {
  title: {
    default: SITE.name,
    template: `%s — ${SITE.name}`,
  },
  description: SITE.description,
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  const publishableKey = process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY
  const clerkAppearance = {
    variables: {
      colorBackground: '#08090b',
      colorInputBackground: '#0d0f12',
      colorPrimary: '#2563ff',
      colorText: '#f0f2f5',
      colorTextSecondary: '#8b95a8',
      borderRadius: '0.375rem',
    },
    elements: {
      card: 'shadow-none border border-[#1c2028]',
      formButtonPrimary: 'bg-[#2563ff] hover:bg-[#1a4acc] text-white text-sm',
    },
  }

  return (
    <html
      lang="en"
      className={`${dmSans.variable} ${jetbrainsMono.variable} bg-ink-950 text-text-primary`}
    >
      <body className="font-body antialiased">
        {clerkEnabled && publishableKey ? (
          <ClerkProvider
            publishableKey={publishableKey}
            appearance={clerkAppearance}
          >
            {children}
          </ClerkProvider>
        ) : (
          children
        )}
      </body>
    </html>
  )
}
