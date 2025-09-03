#!/usr/bin/env node

const fs = require('fs');
const { execSync } = require('child_process');

if (process.argv.length < 3 || (process.argv[2] !== '-l' && process.argv.length < 4)) {
  console.log(`
🔧 TypeScript Fix Tool - sed-like interface

Usage: node sed-fix.js "search_pattern" "replace_pattern" [file1 file2 ...]
       node sed-fix.js -f "filename" "search_pattern" "replace_pattern"
       node sed-fix.js -l                    # List files from files-to-fix.txt

Options:
  -f <file>     Apply fix to specific file
  -l            List all files in files-to-fix.txt with their error counts
  -a            Apply to all files in files-to-fix.txt (default if no files specified)

Examples:
  node sed-fix.js "// Mock cleanup" "// Mock cleanup\\n"
  node sed-fix.js -f "index.test.ts" "\\} expected" "}"
  node sed-fix.js "catch \\(error\\)" "catch (e)"

The tool will:
1. Apply the fix
2. Check TypeScript errors before/after
3. Rollback if errors increase
4. Show success/failure for each file
`);
  process.exit(1);
}

// Get error count functions
function getErrorCount() {
  try {
    const output = execSync('npx tsc --noEmit 2>&1 | grep "error TS" | wc -l', { 
      encoding: 'utf8', stdio: 'pipe'
    });
    return parseInt(output.trim());
  } catch (error) {
    return 9999;
  }
}

function getFileErrors(filePath) {
  try {
    const output = execSync(`npx tsc --noEmit 2>&1 | grep "${filePath}"`, { 
      encoding: 'utf8', stdio: 'pipe'
    });
    return output.split('\n').filter(line => line.includes('error TS')).length;
  } catch (error) {
    return 0;
  }
}

// Get file list
function getFileList() {
  try {
    const content = fs.readFileSync('./files-to-fix.txt', 'utf8');
    return content.split('\n').filter(line => line.trim() && !line.startsWith('#')).map(line => line.trim());
  } catch (error) {
    console.log('⚠️  files-to-fix.txt not found, using all .ts files in e2e/src/suite/api/');
    return [];
  }
}

// Parse arguments
let searchPattern = '';
let replacePattern = '';
let targetFiles = [];
let mode = 'default';

let i = 2;
while (i < process.argv.length) {
  const arg = process.argv[i];
  
  if (arg === '-l') {
    mode = 'list';
    break;
  } else if (arg === '-f') {
    mode = 'single';
    i++;
    if (i < process.argv.length) {
      targetFiles = [process.argv[i]];
    }
  } else if (arg === '-a') {
    mode = 'all';
  } else if (!searchPattern) {
    searchPattern = arg;
  } else if (!replacePattern) {
    replacePattern = arg;
  } else {
    targetFiles.push(arg);
  }
  i++;
}

// Handle list mode
if (mode === 'list') {
  console.log('📋 Files in fix list with error counts:');
  const fileList = getFileList();
  fileList.forEach(filePath => {
    const errors = getFileErrors(filePath);
    const exists = fs.existsSync(filePath) ? '✅' : '❌';
    console.log(`   ${exists} ${filePath} (${errors} errors)`);
  });
  process.exit(0);
}

// Validate required parameters
if (!searchPattern || !replacePattern) {
  console.log('❌ Error: Both search and replace patterns are required');
  process.exit(1);
}

// Determine target files
if (targetFiles.length === 0) {
  targetFiles = getFileList();
}

if (targetFiles.length === 0) {
  console.log('❌ Error: No files to process');
  process.exit(1);
}

console.log(`🔧 Applying sed-like fix to ${targetFiles.length} files...`);
console.log(`📝 Search:  "${searchPattern}"`);
console.log(`📝 Replace: "${replacePattern}"`);

// Convert string patterns to regex (handle basic sed-like patterns)
let searchRegex;
try {
  // Handle basic sed patterns
  const regexPattern = searchPattern
    .replace(/\\n/g, '\n')           // Handle literal \n
    .replace(/\\t/g, '\t')           // Handle literal \t
    .replace(/\\\\/g, '\\');         // Handle escaped backslashes
  
  searchRegex = new RegExp(regexPattern, 'gm');
} catch (error) {
  console.log(`❌ Invalid search pattern: ${error.message}`);
  process.exit(1);
}

// Process replacement string
const replacement = replacePattern
  .replace(/\\n/g, '\n')
  .replace(/\\t/g, '\t')
  .replace(/\\\\/g, '\\');

const initialErrorCount = getErrorCount();
console.log(`📊 Initial total errors: ${initialErrorCount}\n`);

let totalFilesImproved = 0;
let totalErrorsReduced = 0;

// Process each file
targetFiles.forEach((filePath, index) => {
  if (!fs.existsSync(filePath)) {
    console.log(`[${index + 1}/${targetFiles.length}] ❌ File not found: ${filePath}`);
    return;
  }

  const initialFileErrors = getFileErrors(filePath);
  console.log(`[${index + 1}/${targetFiles.length}] 📁 ${filePath} (${initialFileErrors} errors)`);
  
  // Save original content
  const originalContent = fs.readFileSync(filePath, 'utf8');
  
  // Apply the fix
  const newContent = originalContent.replace(searchRegex, replacement);
  
  if (newContent === originalContent) {
    console.log(`   📋 No matches found`);
    return;
  }
  
  // Count matches
  const matches = (originalContent.match(searchRegex) || []).length;
  console.log(`   🔧 Found ${matches} match(es), applying fix...`);
  
  // Write and test
  fs.writeFileSync(filePath, newContent, 'utf8');
  const newFileErrors = getFileErrors(filePath);
  
  if (newFileErrors < initialFileErrors) {
    const reduction = initialFileErrors - newFileErrors;
    console.log(`   ✅ SUCCESS: ${initialFileErrors} → ${newFileErrors} errors (reduced ${reduction})`);
    totalFilesImproved++;
    totalErrorsReduced += reduction;
  } else if (newFileErrors === initialFileErrors) {
    console.log(`   📊 NEUTRAL: Error count unchanged (${newFileErrors})`);
  } else {
    console.log(`   ❌ WORSE: ${initialFileErrors} → ${newFileErrors} errors, reverting`);
    fs.writeFileSync(filePath, originalContent, 'utf8');
  }
});

const finalErrorCount = getErrorCount();
console.log(`\n🎯 SUMMARY:`);
console.log(`   📈 Files improved: ${totalFilesImproved}/${targetFiles.length}`);
console.log(`   📉 Errors reduced: ${totalErrorsReduced}`);
console.log(`   📊 Total: ${initialErrorCount} → ${finalErrorCount} errors`);

// Auto-commit if we have improvements
if (finalErrorCount < initialErrorCount) {
  console.log(`\n🚀 AUTO-COMMITTING ${initialErrorCount - finalErrorCount} error reduction...`);
  
  try {
    execSync('git add .', { stdio: 'pipe' });
    
    const commitMsg = `fix: Reduce ${totalErrorsReduced} TypeScript errors via sed-like fixes

Applied targeted fixes reducing errors from ${initialErrorCount} to ${finalErrorCount}.
Fixed ${totalFilesImproved} file(s) successfully.

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>`;
    
    execSync(`git commit -m "${commitMsg}"`, { stdio: 'pipe' });
    console.log(`   ✅ Committed successfully!`);
  } catch (error) {
    console.log(`   ⚠️  Failed to auto-commit: ${error.message}`);
  }
} else {
  console.log(`\n📋 No overall improvement, but files were safely tested`);
}

// Update file list - remove completed files (0 errors)
console.log(`\n🧹 Updating file list...`);
const allFiles = getFileList();
const remainingFiles = [];
const completedFiles = [];

allFiles.forEach(filePath => {
  if (fs.existsSync(filePath)) {
    const fileErrors = getFileErrors(filePath);
    if (fileErrors === 0) {
      completedFiles.push(filePath);
      console.log(`   ✅ COMPLETED: ${filePath}`);
    } else {
      remainingFiles.push(filePath);
    }
  }
});

if (completedFiles.length > 0) {
  // Update files-to-fix.txt
  const updatedContent = remainingFiles.join('\n') + (remainingFiles.length > 0 ? '\n' : '');
  fs.writeFileSync('./files-to-fix.txt', updatedContent, 'utf8');
  
  // Track completed files
  const completedContent = completedFiles.join('\n') + '\n';
  const existingCompleted = fs.existsSync('./files-completed.txt') ? 
    fs.readFileSync('./files-completed.txt', 'utf8') : '';
  fs.writeFileSync('./files-completed.txt', existingCompleted + completedContent, 'utf8');
  
  console.log(`   📝 Removed ${completedFiles.length} completed files from fix list`);
  console.log(`   📂 ${remainingFiles.length} files remaining`);
}