#!/usr/bin/env node

const fs = require('fs');
const { execSync } = require('child_process');

if (process.argv.length < 4) {
  console.log(`
🔍 Debug Grep Tool - Shows whitespace characters

Usage: node debug-grep.js "pattern" "filename" [context_lines]

This tool helps debug whitespace issues by:
1. Finding lines matching the pattern
2. Showing context lines around matches
3. Converting hidden whitespace to visible characters

Example: node debug-grep.js "catch.*error" "file.ts" 3
`);
  process.exit(1);
}

const pattern = process.argv[2];
const filename = process.argv[3];
const context = process.argv[4] || '2';

if (!fs.existsSync(filename)) {
  console.log(`❌ File not found: ${filename}`);
  process.exit(1);
}

// Convert whitespace to visible characters
function visualizeWhitespace(text) {
  return text
    .replace(/\t/g, '→')      // Tab to arrow
    .replace(/\n/g, '¶\n')    // Newline to paragraph symbol
    .replace(/ /g, '·')       // Space to middle dot
    .replace(/\r/g, '⏎');     // Carriage return
}

try {
  // Use grep with context to find matches
  const grepResult = execSync(`grep -n -A${context} -B${context} "${pattern}" "${filename}"`, { 
    encoding: 'utf8',
    stdio: 'pipe'
  });
  
  console.log(`🔍 Searching for "${pattern}" in ${filename} with ${context} lines context:\n`);
  
  const lines = grepResult.split('\n');
  lines.forEach(line => {
    if (line.trim()) {
      const visualized = visualizeWhitespace(line);
      console.log(visualized);
    }
  });
  
} catch (error) {
  console.log(`❌ No matches found for "${pattern}" in ${filename}`);
}

// Also show TypeScript errors for this file
console.log(`\n📊 TypeScript errors for ${filename}:`);
try {
  const tsErrors = execSync(`npx tsc --noEmit 2>&1 | grep "${filename}"`, { 
    encoding: 'utf8',
    stdio: 'pipe'
  });
  console.log(tsErrors);
} catch (error) {
  console.log('No TypeScript errors found.');
}