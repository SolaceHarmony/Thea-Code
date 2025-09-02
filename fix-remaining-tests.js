#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const glob = require('glob');

function fixTestFile(filePath) {
    let content = fs.readFileSync(filePath, 'utf8');
    let changed = false;
    
    // Fix broken mock syntax left by migration
    // Pattern: "=> {" at start of line indicates broken mock
    if (content.includes('\n=> {') || content.includes('\n=> (')) {
        // Find and fix broken mock blocks
        content = content.replace(/\n=> \{[\s\S]*?\n\}\)/gm, (match) => {
            // This is a broken mock block, comment it out for manual fix
            return '\n// TODO: Fix mock - needs proxyquire\n/*' + match + '*/';
        });
        changed = true;
    }
    
    // Fix remaining expect() patterns
    if (content.includes('expect(')) {
        // toContainEqual -> check if array includes item
        content = content.replace(/expect\(([^)]+)\)\.toContainEqual\(([^)]+)\)/g, 
            'assert.ok($1.some(item => JSON.stringify(item) === JSON.stringify($2)))');
        
        // toBeGreaterThanOrEqual
        content = content.replace(/expect\(([^)]+)\)\.toBeGreaterThanOrEqual\(([^)]+)\)/g, 
            'assert.ok($1 >= $2)');
        
        // toBeLessThanOrEqual
        content = content.replace(/expect\(([^)]+)\)\.toBeLessThanOrEqual\(([^)]+)\)/g, 
            'assert.ok($1 <= $2)');
            
        // expect.stringContaining -> sinon.match for assertions
        content = content.replace(/expect\.stringContaining\(([^)]+)\)/g, 
            'sinon.match.string.and(sinon.match($1))');
            
        // expect.objectContaining -> sinon.match.object
        content = content.replace(/expect\.objectContaining\(([^)]+)\)/g, 
            'sinon.match($1)');
        
        changed = true;
    }
    
    // Fix sinon.spy patterns that need to be sinon.stub
    if (content.includes('sinon.spy(')) {
        // sinon.spy().onFirstCall() doesn't exist - should be stub
        content = content.replace(/sinon\.spy\(([^)]+)\)\.onFirstCall\(\)/g, 
            'sinon.stub($1).onFirstCall()');
        
        // sinon.spy(obj, method).mockImplementation -> sinon.stub
        content = content.replace(/sinon\.spy\(([^,]+),\s*['"]([^'"]+)['"]\)\.mockImplementation/g, 
            'sinon.stub($1, "$2")');
            
        changed = true;
    }
    
    // Fix jest references
    if (content.includes('jest.')) {
        // jest.restoreAllMocks -> sinon.restore
        content = content.replace(/jest\.restoreAllMocks\(\)/g, 'sinon.restore()');
        
        // Remove global jest references
        content = content.replace(/\(globalThis as any\)\.jest = jest/g, 
            '// Global jest reference removed');
            
        changed = true;
    }
    
    // Fix "as jest.Mocked" or "as jest.MockedFunction"
    if (content.includes('as jest.')) {
        content = content.replace(/as jest\.Mocked<[^>]+>/g, '');
        content = content.replace(/as jest\.MockedFunction<[^>]+>/g, '');
        changed = true;
    }
    
    // Fix expect().rejects patterns
    if (content.includes('.rejects.toThrow')) {
        content = content.replace(/await expect\(([^)]+)\)\.rejects\.toThrow\(([^)]+)\)/g, 
            'await assert.rejects($1, $2)');
        content = content.replace(/expect\(([^)]+)\)\.rejects\.toThrow\(([^)]+)\)/g, 
            'await assert.rejects(async () => $1, $2)');
        changed = true;
    }
    
    // Fix type annotations using jest types
    if (content.includes(': jest.')) {
        content = content.replace(/: jest\.SpyInstance/g, ': sinon.SinonSpy');
        content = content.replace(/: jest\.MockedFunction<[^>]+>/g, ': sinon.SinonStub');
        changed = true;
    }
    
    // Ensure file ends with newline
    if (!content.endsWith('\n')) {
        content += '\n';
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
console.log(`Processing ${testFiles.length} test files...`);

let fixedCount = 0;
for (const file of testFiles) {
    if (fixTestFile(file)) {
        console.log(`Fixed: ${path.basename(file)}`);
        fixedCount++;
    }
}

console.log(`\nFixed ${fixedCount} files`);