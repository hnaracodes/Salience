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
      colorBackground: '#FFFFFF',
      colorInputBackground: '#F9F8F3',
      colorPrimary: '#0F0F0F',
      colorText: '#0F0F0F',
      colorTextSecondary: '#5C5C5C',
      borderRadius: '9999px',
    },
    elements: {
      card: 'shadow-card border border-[#E8E6DF]',
      formButtonPrimary: 'bg-[#0F0F0F] hover:bg-[#1A1A1A] text-white text-sm rounded-full',
    },
  }

  return (
    <html
      lang="en"
      className={`${dmSans.variable} ${jetbrainsMono.variable} bg-canvas text-text-primary`}
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
