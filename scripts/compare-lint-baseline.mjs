#!/usr/bin/env node
// Compare current ESLint warnings against a recorded baseline and fail if exceeded.
// Usage: run after producing current JSON via `npm run lint:json`.

import fs from 'node:fs'
import path from 'node:path'

const ROOT = process.cwd()
const BASELINE = path.join(ROOT, 'lint-output', 'eslint-baseline.json')
const CURRENT = path.join(ROOT, 'lint-output', 'current.json')

function readJson(file) {
  try {
    const text = fs.readFileSync(file, 'utf8')
    return JSON.parse(text)
  } catch {
    return null
  }
}

function sumCounts(results) {
  if (!Array.isArray(results)) return { errors: 0, warnings: 0 }
  let errors = 0
  let warnings = 0
  for (const r of results) {
    if (typeof r.errorCount === 'number') errors += r.errorCount
    if (typeof r.warningCount === 'number') warnings += r.warningCount
  }
  return { errors, warnings }
}

const baselineJson = readJson(BASELINE)
if (!baselineJson) {
  console.warn('[compare-lint-baseline] No baseline found at', BASELINE)
  console.warn('  → Run: npm run lint:baseline')
  // Do not fail CI if no baseline yet; treat as pass but visible.
  process.exit(0)
}

const currentJson = readJson(CURRENT)
if (!currentJson) {
  console.error('[compare-lint-baseline] Current lint JSON not found at', CURRENT)
  console.error('  → Run: npm run lint:json')
  process.exit(2)
}

const b = sumCounts(baselineJson)
const c = sumCounts(currentJson)

console.log(`[compare-lint-baseline] Baseline warnings: ${b.warnings}, errors: ${b.errors}`)
console.log(`[compare-lint-baseline] Current  warnings: ${c.warnings}, errors: ${c.errors}`)

if (c.warnings > b.warnings) {
  console.error(`[compare-lint-baseline] Warning count increased by ${c.warnings - b.warnings}. Failing.`)
  process.exit(1)
}

console.log('[compare-lint-baseline] OK: warnings did not exceed baseline.')
process.exit(0)
