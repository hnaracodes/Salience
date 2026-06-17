#!/usr/bin/env node
/**
 * Remove .next using Node fs.rm (handles OneDrive reparse points better than
 * Next's recursive-delete readlink path on Windows).
 */
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const targets = ['.next']

if (process.env.NEXT_DIST_DIR) {
  targets.push(process.env.NEXT_DIST_DIR)
}

for (const name of targets) {
  const dir = path.isAbsolute(name) ? name : path.join(root, name)
  if (!fs.existsSync(dir)) {
    continue
  }
  fs.rmSync(dir, { recursive: true, force: true, maxRetries: 3, retryDelay: 200 })
  console.log(`Removed ${dir}`)
}
