/** @type {import('next').NextConfig} */
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

// Optional: set NEXT_DIST_DIR to a path outside OneDrive on Windows dev machines.
const distDir = process.env.NEXT_DIST_DIR || '.next'

const nextConfig = {
  distDir,
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
  },
  webpack: (config, { dev }) => {
    if (dev) {
      config.cache = {
        type: 'filesystem',
        cacheDirectory: path.join(os.tmpdir(), 'salience-web-webpack-cache'),
        buildDependencies: {
          config: [path.join(__dirname, 'next.config.mjs')],
        },
      }
    }
    return config
  },
}

export default nextConfig
