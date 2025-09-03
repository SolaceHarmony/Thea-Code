#!/usr/bin/env node

/**
 * Surgical syntax fixes for malformed test files
 * Targets only the most critical syntax errors that prevent compilation
 */

const fs = require('fs');
const path = require('path');
const glob = require('glob');

// Track what we fix
const fixLog = {
    files: [],
    totalFixes: 0,
    fixTypes: {}
};

/**
 * Priority 1: Fix malformed imports that break syntax parsing
 */
function fixCriticalImportErrors(content, filePath) {
    let fixed = content;
    let changeCount = 0;
    
    // 1. Fix import statement directly followed by comment block (no newline)
    // import * as sinon from 'sinon'/** -> import * as sinon from 'sinon'\n/**
    const importCommentRegex = /^(import\s+(?:\*\s+as\s+\w+|\{[^}]+\}|\w+)\s+from\s+['"][^'"]+['"])(\/\*\*)/gm;
    if (importCommentRegex.test(fixed)) {
        fixed = fixed.replace(importCommentRegex, '$1\n$2');
        changeCount++;
        console.log(`  ✓ Fixed import-comment fusion in ${path.basename(filePath)}`);
    }
    
    // 2. Fix import inside another import's braces
    // import {\nimport * as assert -> import * as assert\nimport {
    const nestedImportRegex = /^(import\s+\{[^\}]*?)\n(import\s+(?:\*\s+as\s+\w+|\w+)\s+from\s+['"][^'"]+['"])/gm;
    if (nestedImportRegex.test(fixed)) {
        fixed = fixed.replace(nestedImportRegex, '$2\n$1');
        changeCount++;
        console.log(`  ✓ Fixed nested import in ${path.basename(filePath)}`);
    }
    
    // 3. Fix broken import with missing 'from'
    // import * as assert 'assert' -> import * as assert from 'assert'
    fixed = fixed.replace(/^(import\s+\*\s+as\s+\w+)\s+(['"][^'"]+['"])$/gm, '$1 from $2');
    
    // 4. Fix malformed destructured imports on wrong line
    // import {\nimport * as X from 'x'\n  A, B\n} from 'y'
    const lines = fixed.split('\n');
    const fixedLines = [];
    let i = 0;
    
    while (i < lines.length) {
        const line = lines[i];
        
        // Check for import { followed by another import
        if (line.match(/^import\s+\{/) && i + 1 < lines.length) {
            const nextLine = lines[i + 1];
            if (nextLine.match(/^import\s+/)) {
                // Move the complete import first
                fixedLines.push(nextLine);
                fixedLines.push(line);
                i += 2;
                changeCount++;
                continue;
            }
        }
        
        fixedLines.push(line);
        i++;
    }
    
    if (changeCount > 0) {
        fixed = fixedLines.join('\n');
    }
    
    fixLog.totalFixes += changeCount;
    return fixed;
}

/**
 * Priority 2: Fix malformed TODO comments that break proxyquire patterns
 */
function fixMalformedTodoComments(content, filePath) {
    let fixed = content;
    let changeCount = 0;
    
    // Fix TODO comments that bleed into code
    // // TODO: Use proxyquire for module mocking - "@aws-sdk/client-bedrock-runtime", () => ({
    // becomes proper comment
    const todoCodeRegex = /\/\/\s*TODO:\s*Use proxyquire[^-]*-\s*("[^"]+"),\s*\(\)\s*=>\s*\(\{/g;
    
    if (todoCodeRegex.test(fixed)) {
        fixed = fixed.replace(todoCodeRegex, (match, moduleName) => {
            changeCount++;
            return `// TODO: Use proxyquire for module mocking\n\t\t// Mock for ${moduleName} needed here`;
        });
        console.log(`  ✓ Fixed malformed TODO-code fusion in ${path.basename(filePath)}`);
    }
    
    fixLog.totalFixes += changeCount;
    return fixed;
}

/**
 * Priority 3: Fix critical closing bracket issues
 */
function fixBracketIssues(content, filePath) {
    let fixed = content;
    let changeCount = 0;
    
    // Fix floating })) that should be })
    // Common after malformed mock setups
    const lines = fixed.split('\n');
    const fixedLines = [];
    
    for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
        const trimmed = line.trim();
        
        // Look for problematic patterns
        if (trimmed === '}))' && i > 0) {
            // Check context - if previous line ends with }, this is likely wrong
            const prevLine = lines[i - 1].trim();
            if (prevLine.endsWith('},') || prevLine.endsWith('}')) {
                fixedLines.push(line.replace('})', '}'));
                changeCount++;
                continue;
            }
        }
        
        fixedLines.push(line);
    }
    
    if (changeCount > 0) {
        fixed = fixedLines.join('\n');
        console.log(`  ✓ Fixed ${changeCount} bracket issues in ${path.basename(filePath)}`);
        fixLog.totalFixes += changeCount;
    }
    
    return fixed;
}

/**
 * Priority 4: Fix the most broken expect() statements only
 */
function fixCriticalExpectStatements(content, filePath) {
    let fixed = content;
    let changeCount = 0;
    
    // Only fix expect statements that cause syntax errors
    // Focus on malformed ones like expect()with() or expect().called)With
    
    // Fix: .called)With -> .calledWith
    fixed = fixed.replace(/\.called\)With/g, '.calledWith');
    
    // Fix: expect().not.property)
    fixed = fixed.replace(/expect\([^)]+\)\.not\.(\w+)\)/g, (match, prop) => {
        changeCount++;
        return `assert.ok(!/* CHECK: not.${prop} */)`;
    });
    
    // Fix the most common and safe conversions only
    const conversions = [
        [/expect\(([^)]+)\)\.toBe\(([^)]+)\)/g, 'assert.strictEqual($1, $2)'],
        [/expect\(([^)]+)\)\.toEqual\(([^)]+)\)/g, 'assert.deepStrictEqual($1, $2)'],
        [/expect\(([^)]+)\)\.toBeTruthy\(\)/g, 'assert.ok($1)'],
        [/expect\(([^)]+)\)\.toBeFalsy\(\)/g, 'assert.ok(!$1)'],
        [/expect\(([^)]+)\)\.toBeUndefined\(\)/g, 'assert.strictEqual($1, undefined)'],
        [/expect\(([^)]+)\)\.toStrictEqual\(([^)]+)\)/g, 'assert.deepStrictEqual($1, $2)']
    ];
    
    for (const [pattern, replacement] of conversions) {
        const before = fixed;
        fixed = fixed.replace(pattern, replacement);
        if (before !== fixed) {
            changeCount++;
        }
    }
    
    if (changeCount > 0) {
        console.log(`  ✓ Fixed ${changeCount} expect statements in ${path.basename(filePath)}`);
        fixLog.totalFixes += changeCount;
    }
    
    return fixed;
}

/**
 * Apply only the most critical fixes to a file
 */
function applySurgicalFixes(filePath) {
    try {
        let content = fs.readFileSync(filePath, 'utf8');
        const original = content;
        
        // Apply fixes in priority order
        content = fixCriticalImportErrors(content, filePath);
        content = fixMalformedTodoComments(content, filePath);
        content = fixBracketIssues(content, filePath);
        content = fixCriticalExpectStatements(content, filePath);
        
        if (content !== original) {
            fs.writeFileSync(filePath, content);
            fixLog.files.push(filePath);
            return true;
        }
        
        return false;
    } catch (error) {
        console.error(`  ✗ Error processing ${filePath}: ${error.message}`);
        return false;
    }
}

// Main execution
console.log('🔧 Applying surgical syntax fixes...\n');
console.log('Targeting only critical syntax errors that prevent compilation.\n');

// Focus on files with the worst syntax errors first
const criticalPatterns = [
    'e2e/src/suite/api/providers/*.test.ts',
    'e2e/src/suite/services/mcp/**/*.test.ts', 
    'e2e/src/suite/core/**/*.test.ts',
    'e2e/src/suite/utils/*.test.ts'
];

let totalFixed = 0;

for (const pattern of criticalPatterns) {
    const files = glob.sync(pattern);
    
    if (files.length === 0) continue;
    
    console.log(`\nProcessing ${files.length} files matching ${pattern.replace('e2e/src/suite/', '')}:`);
    
    for (const file of files) {
        if (applySurgicalFixes(file)) {
            totalFixed++;
        }
    }
}

// Summary
console.log('\n' + '='.repeat(60));
console.log('✅ SURGICAL FIX COMPLETE\n');
console.log(`Files modified: ${fixLog.files.length}`);
console.log(`Total fixes applied: ${fixLog.totalFixes}`);

if (fixLog.files.length > 0) {
    console.log('\nSample files fixed:');
    fixLog.files.slice(0, 10).forEach(f => {
        console.log(`  - ${f.replace(/^.*\/e2e\//, 'e2e/')}`);
    });
}

console.log('\n📝 Next step: Run TypeScript compiler to verify fixes');
console.log('   npx tsc --noEmit 2>&1 | grep -c "error TS"');