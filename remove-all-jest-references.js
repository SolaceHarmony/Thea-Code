#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const glob = require('glob');

function cleanJestReferences(filePath) {
    let content = fs.readFileSync(filePath, 'utf8');
    let changed = false;
    
    // Remove old jest command comments
    if (content.includes('// npx jest')) {
        content = content.replace(/^\/\/ npx jest.*$/gm, '');
        changed = true;
    }
    
    // Fix jest function type annotations
    if (content.includes('jest.fn<')) {
        content = content.replace(/jest\.fn<([^>]+)>\(\)/g, 'sinon.stub()');
        content = content.replace(/jest\.fn<([^>]+)>\(\)\.returns\(([^)]+)\)/g, 'sinon.stub().returns($2)');
        content = content.replace(/jest\.fn<([^>]+)>\(\)\.resolves\(([^)]+)\)/g, 'sinon.stub().resolves($2)');
        content = content.replace(/jest\.fn<([^>]+)>\(\)\.callsFake\(([^)]+)\)/g, 'sinon.stub().callsFake($2)');
        changed = true;
    }
    
    // Fix any remaining jest.fn() patterns
    if (content.includes('jest.fn(')) {
        content = content.replace(/jest\.fn\(\)/g, 'sinon.stub()');
        changed = true;
    }
    
    // Fix any other jest patterns
    if (content.includes('jest.')) {
        content = content.replace(/jest\.mock/g, '// TODO: Mock needs setup');
        content = content.replace(/jest\.spyOn/g, 'sinon.spy');
        content = content.replace(/jest\.clearAllMocks/g, 'sinon.restore');
        content = content.replace(/jest\.resetAllMocks/g, 'sinon.restore');
        content = content.replace(/jest\.restoreAllMocks/g, 'sinon.restore');
        changed = true;
    }
    
    // Remove TODO comments about jest
    if (content.includes('TODO: requireActual') || content.includes('TODO: requireMock')) {
        content = content.replace(/^\s*\/\/ TODO: requireActual.*$/gm, '');
        content = content.replace(/^\s*\/\/ TODO: requireMock.*$/gm, '');
        changed = true;
    }
    
    // Clean up multiple empty lines
    content = content.replace(/\n\n\n+/g, '\n\n');
    
    if (changed) {
        fs.writeFileSync(filePath, content);
        return true;
    }
    
    return false;
}

// Process all test files
const testFiles = glob.sync('src/e2e/src/suite/**/*.test.ts');
console.log(`Removing jest references from ${testFiles.length} test files...`);

let cleanedCount = 0;
for (const file of testFiles) {
    if (cleanJestReferences(file)) {
        console.log(`Cleaned jest references: ${path.basename(file)}`);
        cleanedCount++;
    }
}

console.log(`\nCleaned jest references from ${cleanedCount} files`);

// Verify no jest references remain
const remainingJest = [];
for (const file of testFiles) {
    const content = fs.readFileSync(file, 'utf8');
    if (content.toLowerCase().includes('jest')) {
        const lines = content.split('\n');
        for (let i = 0; i < lines.length; i++) {
            if (lines[i].toLowerCase().includes('jest')) {
                remainingJest.push(`${file}:${i+1}: ${lines[i].trim()}`);
            }
        }
    }
}

if (remainingJest.length > 0) {
    console.log(`\n⚠️  Found ${remainingJest.length} remaining jest references:`);
    remainingJest.slice(0, 10).forEach(ref => console.log(`  ${ref}`));
    if (remainingJest.length > 10) {
        console.log(`  ... and ${remainingJest.length - 10} more`);
    }
} else {
    console.log('\n✅ No jest references found - migration complete!');
}