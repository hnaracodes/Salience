import type { Config } from 'tailwindcss'

const config: Config = {
  content: ['./app/**/*.{ts,tsx}', './components/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        canvas: {
          DEFAULT: '#F9F8F3',
          warm: '#F3F1EA',
        },
        surface: {
          DEFAULT: '#FFFFFF',
          muted: '#F5F4EF',
          hover: '#EFEDE6',
        },
        line: {
          DEFAULT: '#E8E6DF',
          strong: '#D4D2CA',
        },
        ink: {
          950: '#0F0F0F',
          900: '#1A1A1A',
          800: '#2D2D2D',
          700: '#404040',
          600: '#6B6B6B',
        },
        signal: {
          DEFAULT: '#1D4ED8',
          dim: '#1E3A8A',
          glow: 'rgba(29,78,216,0.12)',
        },
        neural: '#E85D24',
        copy: '#15803D',
        muted: '#737373',
        text: {
          primary: '#0F0F0F',
          secondary: '#5C5C5C',
          tertiary: '#8A8A8A',
        },
      },
      fontFamily: {
        display: ['var(--font-dm-sans)', 'system-ui', 'sans-serif'],
        body: ['var(--font-dm-sans)', 'system-ui', 'sans-serif'],
        mono: ['var(--font-jetbrains-mono)', 'monospace'],
      },
      borderRadius: {
        pill: '9999px',
        card: '1.25rem',
        'card-lg': '1.5rem',
      },
      boxShadow: {
        nav: '0 2px 20px rgba(15,15,15,0.06), 0 1px 3px rgba(15,15,15,0.04)',
        card: '0 4px 24px -4px rgba(15,15,15,0.08), 0 1px 3px rgba(15,15,15,0.04)',
        'card-lg':
          '0 20px 60px -16px rgba(15,15,15,0.14), 0 8px 24px -8px rgba(15,15,15,0.08)',
        'card-float':
          '0 32px 64px -20px rgba(15,15,15,0.18), 0 12px 32px -12px rgba(15,15,15,0.1)',
        product: '0 24px 48px -12px rgba(15,15,15,0.22)',
      },
      animation: {
        'pulse-signal': 'pulse-signal 2.5s ease-in-out infinite',
        'float-gentle': 'float-gentle 6s ease-in-out infinite',
      },
      keyframes: {
        'pulse-signal': {
          '0%, 100%': { opacity: '0.5' },
          '50%': { opacity: '1' },
        },
        'float-gentle': {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-8px)' },
        },
      },
      transitionTimingFunction: {
        premium: 'cubic-bezier(0.22, 1, 0.36, 1)',
      },
    },
  },
  plugins: [],
}

export default config
