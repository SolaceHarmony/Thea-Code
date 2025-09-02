#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const glob = require('glob');

function fixMigratedFile(filePath) {
    console.log(`Fixing: ${filePath}`);
    
    let content = fs.readFileSync(filePath, 'utf8');
    const originalContent = content;
    
    // Fix broken jest.mock patterns that got mangled
    // Pattern: "=> ({" or "=> (" at the start of a line is a broken mock
    content = content.replace(/^=>\s*\(\{/gm, '// TODO: Mock needs manual migration\n// Original mock: {');
    content = content.replace(/^=>\s*\(/gm, '// TODO: Mock needs manual migration\n// ');
    
    // Fix duplicate imports (vscode imported twice, etc)
    const lines = content.split('\n');
    const seenImports = new Set();
    const fixedLines = [];
    
    for (const line of lines) {
        // Skip duplicate import lines
        if (line.startsWith('import')) {
            const importKey = line.trim();
            if (seenImports.has(importKey)) {
                continue; // Skip duplicate
            }
            seenImports.add(importKey);
        }
        fixedLines.push(line);
    }
    
    content = fixedLines.join('\n');
    
    // Fix broken type imports where assert/sinon got injected mid-import
    content = content.replace(/import type \{[\s\S]*?import \* as assert[\s\S]*?import \* as sinon[\s\S]*?\}/gm, (match) => {
        // Extract the original type import content
        const typeMatch = match.match(/import type \{([^}]*?)import \* as assert/s);
        if (typeMatch) {
            const types = typeMatch[1].trim();
            // Find the rest after sinon import
            const afterMatch = match.match(/import \* as sinon from 'sinon'\s*(.*?)\}/s);
            const afterTypes = afterMatch ? afterMatch[1].trim() : '';
            
            // Reconstruct properly
            return `import type {\n  ${types}\n  ${afterTypes}\n}\nimport * as assert from 'assert'\nimport * as sinon from 'sinon'`;
        }
        return match;
    });
    
    // Fix malformed mock patterns from Jest
    content = content.replace(/jest\.MockedFunction<any>/g, 'sinon.SinonStub');
    content = content.replace(/jest\.MockedFunction<[^>]+>/g, 'sinon.SinonStub');
    content = content.replace(/jest\.Mock/g, 'sinon.SinonStub');
    content = content.replace(/jest\.resetAllMocks\(\)/g, 'sinon.restore()');
    
    // Fix expect patterns that didn't convert properly
    content = content.replace(/expect\(([^)]+)\)\.not\.toHaveBeenCalled\(\)/g, 'assert.ok(!$1.called)');
    content = content.replace(/expect\(([^)]+)\)\.toHaveBeenCalled\(\)/g, 'assert.ok($1.called)');
    
    // Remove standalone colon lines (broken syntax)
    content = content.replace(/^\s*:\s*$/gm, '');
    
    // Fix "No newline at end of file" appearing mid-file
    content = content.replace(/\n\s*No newline at end of file\s*\n/g, '\n');
    
    // Ensure file ends with newline
    if (!content.endsWith('\n')) {
        content += '\n';
    }
    
    if (content !== originalContent) {
        fs.writeFileSync(filePath, content);
        console.log(`  ✓ Fixed`);
        return true;
    }
    
    console.log(`  - No changes needed`);
    return false;
}

// Find all e2e test files
const testFiles = glob.sync('src/e2e/src/suite/**/*.e2e.test.ts');

console.log(`Found ${testFiles.length} test files to check`);

let fixedCount = 0;
for (const file of testFiles) {
    if (fixMigratedFile(file)) {
        fixedCount++;
    }
}

console.log(`\nFixed ${fixedCount} files`);