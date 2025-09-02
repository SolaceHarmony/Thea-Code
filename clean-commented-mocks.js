#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const glob = require('glob');

function cleanTestFile(filePath) {
    let content = fs.readFileSync(filePath, 'utf8');
    let changed = false;
    
    // Fix broken mock syntax in commented blocks
    if (content.includes('// Original mock: {')) {
        // Remove broken mock blocks that are partially commented
        content = content.replace(/\/\/ Original mock: \{[\s\S]*?\}\)\)/gm, '// TODO: Implement proper mock with proxyquire');
        
        // Remove stray }) and )) patterns
        content = content.replace(/^\s*\}\)\)?\s*$/gm, '');
        
        // Remove standalone TODO: requireActual lines
        content = content.replace(/^\s*\/\/ TODO: requireActual.*$/gm, '');
        
        // Remove lines with just "..." and TODO comments
        content = content.replace(/^\s*\.\.\.\/\/ TODO:.*$/gm, '');
        
        changed = true;
    }
    
    // Remove broken lines with just braces or parentheses
    if (content.match(/^\s*[\}\)\]\,]\s*$/m)) {
        content = content.replace(/^\s*[\}\)\]\,]\s*$/gm, '');
        changed = true;
    }
    
    // Remove empty lines between imports and fix import organization
    const lines = content.split('\n');
    const cleanedLines = [];
    let inImportSection = false;
    let lastLineWasEmpty = false;
    
    for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
        const trimmed = line.trim();
        
        // Track if we're in import section
        if (trimmed.startsWith('import ') || trimmed.startsWith('export ')) {
            inImportSection = true;
        } else if (inImportSection && trimmed && !trimmed.startsWith('//') && !trimmed.startsWith('/*')) {
            inImportSection = false;
        }
        
        // Skip multiple empty lines
        if (trimmed === '') {
            if (!lastLineWasEmpty) {
                cleanedLines.push(line);
                lastLineWasEmpty = true;
            }
            continue;
        }
        
        // Skip broken syntax lines
        if (trimmed === '}))' || trimmed === '})' || trimmed === '))' || 
            trimmed === '}' || trimmed === ')' || trimmed === '];') {
            continue;
        }
        
        cleanedLines.push(line);
        lastLineWasEmpty = false;
    }
    
    const newContent = cleanedLines.join('\n');
    if (newContent !== content) {
        changed = true;
        content = newContent;
    }
    
    // Ensure file ends with single newline
    if (!content.endsWith('\n')) {
        content += '\n';
        changed = true;
    } else if (content.endsWith('\n\n\n')) {
        content = content.replace(/\n+$/, '\n');
        changed = true;
    }
    
    if (changed) {
        fs.writeFileSync(filePath, content);
        return true;
    }
    
    return false;
}

// Process all test files
const testFiles = glob.sync('src/e2e/src/suite/**/*.test.ts');
console.log(`Cleaning ${testFiles.length} test files...`);

let cleanedCount = 0;
for (const file of testFiles) {
    if (cleanTestFile(file)) {
        console.log(`Cleaned: ${path.basename(file)}`);
        cleanedCount++;
    }
}

console.log(`\nCleaned ${cleanedCount} files`);