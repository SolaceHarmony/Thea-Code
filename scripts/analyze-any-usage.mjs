#!/usr/bin/env node
/**
 * Analyze TypeScript "any" usage across the repo and print counts.
 *
 * Usage:
 *   node scripts/analyze-any-usage.mjs
 *
 * Output:
 *   - Human-readable summary to stdout
 *   - JSON summary written to lint-output/any-usage.json
 */

import fs from 'node:fs'
import path from 'node:path'
import { globby } from 'globby'

const ROOT = process.cwd()
const OUT_DIR = path.join(ROOT, 'lint-output')
const OUT_FILE = path.join(OUT_DIR, 'any-usage.json')

function countAnyInText(text) {
  // Rough, fast heuristic: count bare word "any" occurrences, excluding within quotes.
  // This will still count type references like Array<any>, Promise<any>, etc., which is desired.
  // We skip matches inside // and /* */ comments to avoid false positives in comments.
  // Simple approach: strip comments/strings with regex-best-effort, not a full parser.

  // Remove block comments
  let code = text.replace(/\/\*[\s\S]*?\*\//g, '')
  // Remove line comments
  code = code.replace(/(^|\s)\/\/.*$/gm, '$1')
  // Remove single and double quoted strings and template strings
  code = code.replace(/(['"`])(?:(?!\1)[\\\s\S])*\1/g, '""')

  const matches = code.match(/\bany\b/g)
  return matches ? matches.length : 0
}

async function main() {
  const include = [
    'src/**/*.{ts,tsx}',
    'webview-ui/**/*.{ts,tsx}',
    '!**/*.d.ts',
    '!**/node_modules/**',
    '!dist/**',
    '!build/**',
    '!coverage/**',
    '!coverage-report/**',
    '!webview-ui/dist/**',
    '!webview-ui/build/**',
  ]

  const files = await globby(include, { gitignore: true, expandDirectories: false })

  let total = 0
  const perFile = []

  for (const f of files) {
    try {
      const text = fs.readFileSync(f, 'utf8')
      const count = countAnyInText(text)
      if (count > 0) {
        perFile.push({ file: f, count })
        total += count
      }
    } catch (err) {
      // ignore read errors
    }
  }

  // Sort descending by count
  perFile.sort((a, b) => b.count - a.count)

  const summary = {
    generatedAt: new Date().toISOString(),
    totalAnyCount: total,
    filesWithAny: perFile.length,
    top: perFile.slice(0, 50),
  }

  if (!fs.existsSync(OUT_DIR)) fs.mkdirSync(OUT_DIR, { recursive: true })
  fs.writeFileSync(OUT_FILE, JSON.stringify(summary, null, 2), 'utf8')

  console.log(`[analyze-any-usage] Files scanned: ${files.length}`)
  console.log(`[analyze-any-usage] Files with any: ${perFile.length}`)
  console.log(`[analyze-any-usage] Total \"any\" occurrences: ${total}`)
  console.log(`[analyze-any-usage] Wrote summary → ${path.relative(ROOT, OUT_FILE)}`)
}

main().catch((err) => {
  console.error('[analyze-any-usage] Failed:', err && err.message ? err.message : err)
  process.exit(1)
})
