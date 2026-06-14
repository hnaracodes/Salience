import type { Config } from 'tailwindcss'

const config: Config = {
  content: ['./app/**/*.{ts,tsx}', './components/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        ink: {
          950: '#08090b',
          900: '#0d0f12',
          800: '#13161c',
          700: '#1c2028',
          600: '#262c38',
        },
        signal: {
          DEFAULT: '#2563ff',
          dim: '#1a4acc',
          glow: 'rgba(37,99,255,0.15)',
        },
        neural: '#ff6b35',
        copy: '#22c55e',
        muted: '#6b7280',
        text: {
          primary: '#f0f2f5',
          secondary: '#8b95a8',
        },
      },
      fontFamily: {
        display: ['"DM Sans"', 'sans-serif'],
        body: ['"DM Sans"', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'monospace'],
      },
      animation: {
        'pulse-signal': 'pulse-signal 2s ease-in-out infinite',
      },
      keyframes: {
        'pulse-signal': {
          '0%, 100%': { opacity: '0.4' },
          '50%': { opacity: '1' },
        },
      },
    },
  },
  plugins: [],
}

export default config
