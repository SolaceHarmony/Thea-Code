#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

function migrateTestFile(filePath) {
    let content = fs.readFileSync(filePath, 'utf8');
    const originalContent = content;
    
    // Track if we need sinon
    const needsSinon = content.includes('jest.fn()') || 
                       content.includes('jest.mock') || 
                       content.includes('jest.spyOn') ||
                       content.includes('.mock') ||
                       content.includes('Mock');
    
    // Basic test structure replacements
    content = content.replace(/describe\(/g, 'suite(');
    content = content.replace(/\bit\(/g, 'test(');
    content = content.replace(/\bit\.skip\(/g, 'test.skip(');
    content = content.replace(/\bit\.only\(/g, 'test.only(');
    content = content.replace(/beforeEach\(/g, 'setup(');
    content = content.replace(/afterEach\(/g, 'teardown(');
    content = content.replace(/beforeAll\(/g, 'suiteSetup(');
    content = content.replace(/afterAll\(/g, 'suiteTeardown(');
    
    // Assertion replacements
    content = content.replace(/expect\(([^)]+)\)\.toBe\(([^)]+)\)/g, 'assert.strictEqual($1, $2)');
    content = content.replace(/expect\(([^)]+)\)\.toEqual\(([^)]+)\)/g, 'assert.deepStrictEqual($1, $2)');
    content = content.replace(/expect\(([^)]+)\)\.toStrictEqual\(([^)]+)\)/g, 'assert.deepStrictEqual($1, $2)');
    content = content.replace(/expect\(([^)]+)\)\.toBeTruthy\(\)/g, 'assert.ok($1)');
    content = content.replace(/expect\(([^)]+)\)\.toBeFalsy\(\)/g, 'assert.ok(!$1)');
    content = content.replace(/expect\(([^)]+)\)\.toBeUndefined\(\)/g, 'assert.strictEqual($1, undefined)');
    content = content.replace(/expect\(([^)]+)\)\.toBeNull\(\)/g, 'assert.strictEqual($1, null)');
    content = content.replace(/expect\(([^)]+)\)\.toBeDefined\(\)/g, 'assert.ok($1 !== undefined)');
    content = content.replace(/expect\(([^)]+)\)\.toContain\(([^)]+)\)/g, 'assert.ok($1.includes($2))');
    content = content.replace(/expect\(([^)]+)\)\.toHaveLength\(([^)]+)\)/g, 'assert.strictEqual($1.length, $2)');
    content = content.replace(/expect\(([^)]+)\)\.toBeInstanceOf\(([^)]+)\)/g, 'assert.ok($1 instanceof $2)');
    content = content.replace(/expect\(([^)]+)\)\.toMatch\(([^)]+)\)/g, 'assert.ok($1.match($2))');
    content = content.replace(/expect\(([^)]+)\)\.toThrow\(\)/g, 'assert.throws(() => $1)');
    content = content.replace(/expect\(([^)]+)\)\.not\.toBe\(([^)]+)\)/g, 'assert.notStrictEqual($1, $2)');
    content = content.replace(/expect\(([^)]+)\)\.not\.toEqual\(([^)]+)\)/g, 'assert.notDeepStrictEqual($1, $2)');
    
    // Handle toBeCloseTo for floating point comparisons
    content = content.replace(/expect\(([^)]+)\)\.toBeCloseTo\(([^,]+),\s*(\d+)\)/g, 
        'assert.ok(Math.abs($1 - $2) < Math.pow(10, -$3))');
    
    // Handle toHaveBeenCalled patterns
    content = content.replace(/expect\(([^)]+)\)\.toHaveBeenCalled\(\)/g, 'assert.ok($1.called)');
    content = content.replace(/expect\(([^)]+)\)\.toHaveBeenCalledTimes\(([^)]+)\)/g, 'assert.strictEqual($1.callCount, $2)');
    content = content.replace(/expect\(([^)]+)\)\.toHaveBeenCalledWith\(([^)]+)\)/g, 'assert.ok($1.calledWith($2))');
    
    // Mock replacements
    content = content.replace(/jest\.fn\(\)/g, 'sinon.stub()');
    content = content.replace(/jest\.fn\(([^)]+)\)/g, 'sinon.stub().returns($1)');
    content = content.replace(/jest\.spyOn\(([^,]+),\s*([^)]+)\)/g, 'sinon.spy($1, $2)');
    content = content.replace(/\.mockReturnValue\(([^)]+)\)/g, '.returns($1)');
    content = content.replace(/\.mockResolvedValue\(([^)]+)\)/g, '.resolves($1)');
    content = content.replace(/\.mockRejectedValue\(([^)]+)\)/g, '.rejects($1)');
    content = content.replace(/\.mockImplementation\(([^)]+)\)/g, '.callsFake($1)');
    
    // Remove jest.mock calls (need manual handling)
    content = content.replace(/jest\.mock\([^)]+\);?\s*/g, '// TODO: Mock setup needs manual migration\n');
    
    // Fix imports
    let imports = [];
    
    // Always add assert
    imports.push("import * as assert from 'assert'");
    
    // Add sinon if needed
    if (needsSinon) {
        imports.push("import * as sinon from 'sinon'");
    }
    
    // Add vscode if it's imported
    if (content.includes('vscode')) {
        imports.push("import * as vscode from 'vscode'");
    }
    
    // Remove old jest-related imports
    content = content.replace(/import.*['"]jest.*['"];?\s*/g, '');
    content = content.replace(/import.*@jest.*\s*/g, '');
    
    // Add new imports at the top
    const importIndex = content.indexOf('import');
    if (importIndex >= 0) {
        // Find the first non-import line
        const lines = content.split('\n');
        let firstNonImportIndex = 0;
        for (let i = 0; i < lines.length; i++) {
            if (lines[i].trim() && !lines[i].startsWith('import') && !lines[i].startsWith('//')) {
                firstNonImportIndex = i;
                break;
            }
        }
        
        // Insert our imports
        lines.splice(firstNonImportIndex, 0, ...imports);
        content = lines.join('\n');
    } else {
        content = imports.join('\n') + '\n\n' + content;
    }
    
    // Fix any remaining test.each patterns (these need manual work)
    if (content.includes('.each')) {
        content = '// TODO: test.each patterns need manual migration\n' + content;
    }
    
    // Clean up double imports
    const uniqueImports = new Set();
    content = content.split('\n').filter(line => {
        if (line.startsWith('import')) {
            if (uniqueImports.has(line)) {
                return false;
            }
            uniqueImports.add(line);
        }
        return true;
    }).join('\n');
    
    return content;
}

function processFile(filePath, destPath) {
    console.log(`Processing: ${filePath}`);
    
    try {
        const content = migrateTestFile(filePath);
        
        // Use provided destination path or default to .e2e.test.ts
        const newPath = destPath || filePath.replace('.test.ts', '.e2e.test.ts');
        
        // Ensure directory exists
        const dir = path.dirname(newPath);
        if (!fs.existsSync(dir)) {
            fs.mkdirSync(dir, { recursive: true });
        }
        
        fs.writeFileSync(newPath, content);
        console.log(`  ✓ Migrated to: ${newPath}`);
        
        // Optionally remove old file
        // fs.unlinkSync(filePath);
        
    } catch (error) {
        console.error(`  ✗ Error: ${error.message}`);
    }
}

// Process files from command line or find all test files
const args = process.argv.slice(2);

if (args.length > 0) {
    // Check if we have source and destination paths
    if (args.length === 2 && fs.existsSync(args[0])) {
        // Process single file with destination
        processFile(args[0], args[1]);
    } else {
        // Process multiple files without specific destinations
        args.forEach(file => {
            if (fs.existsSync(file)) {
                processFile(file);
            } else {
                console.error(`File not found: ${file}`);
            }
        });
    }
} else {
    console.log('Usage: node migrate-jest-to-mocha.js <source-file> [dest-file]');
    console.log('Or: node migrate-jest-to-mocha.js <test-file1> <test-file2> ...');
}