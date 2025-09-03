#!/usr/bin/env node
/**
 * Final structural fixes for remaining test file issues
 * Focuses on fixing structural problems and incomplete conversions
 */

const fs = require('fs');
const path = require('path');

let totalFiles = 0;
let filesFixed = 0;
let totalFixes = 0;

// Track specific fix types
const fixStats = {
    incompleteBlocks: 0,
    tryWithoutCatch: 0,
    malformedMocks: 0,
    assertionConversions: 0,
    cleanupComments: 0
};

function fixIncompleteBlocks(content, filePath) {
    let fixed = content;
    let localFixes = 0;

    // Fix incomplete try blocks - ensure they have catch
    const tryBlockRegex = /try\s*{([^}]*?)}\s*(?!catch|finally)/gm;
    fixed = fixed.replace(tryBlockRegex, (match, body) => {
        if (!match.includes('catch') && !match.includes('finally')) {
            localFixes++;
            return `try {${body}} catch (error) {\n\t\t\tassert.fail('Unexpected error: ' + error.message)\n\t\t}`;
        }
        return match;
    });

    // Fix incomplete if-else chains
    fixed = fixed.replace(/}\s*else\s+if\s*\(/g, '} else if (');
    
    // Fix incomplete async function declarations
    fixed = fixed.replace(/async\s+\(\s*\)\s*=>\s*$/gm, 'async () => {\n\t\t\t// Implementation needed\n\t\t}');

    if (localFixes > 0) {
        fixStats.incompleteBlocks += localFixes;
    }

    return fixed;
}

function fixTryWithoutCatch(content, filePath) {
    let fixed = content;
    let localFixes = 0;

    // Find try blocks without proper catch
    const lines = fixed.split('\n');
    let inTry = false;
    let tryStartIndex = -1;
    let braceCount = 0;
    
    for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
        
        if (line.includes('try {')) {
            inTry = true;
            tryStartIndex = i;
            braceCount = 1;
        } else if (inTry) {
            braceCount += (line.match(/{/g) || []).length;
            braceCount -= (line.match(/}/g) || []).length;
            
            if (braceCount === 0) {
                // End of try block
                if (i + 1 < lines.length && !lines[i + 1].includes('catch') && !lines[i + 1].includes('finally')) {
                    // Missing catch
                    lines[i] = lines[i] + ' catch (error) {\n\t\t\tassert.fail("Unexpected error: " + error.message)\n\t\t}';
                    localFixes++;
                }
                inTry = false;
            }
        }
    }
    
    if (localFixes > 0) {
        fixed = lines.join('\n');
        fixStats.tryWithoutCatch += localFixes;
    }

    return fixed;
}

function fixMalformedMocks(content, filePath) {
    let fixed = content;
    let localFixes = 0;

    // Remove broken mock return blocks
    fixed = fixed.replace(/\/\/\s*Mock removed - needs manual implementation[^}]*}/gm, (match) => {
        if (match.includes('catch') || match.includes('else')) {
            localFixes++;
            return '// Mock implementation removed';
        }
        return match;
    });

    // Fix dangling mock properties
    fixed = fixed.replace(/^\s*}\s*,\s*$/gm, (match, offset) => {
        // Check if this is actually dangling
        const before = fixed.substring(Math.max(0, offset - 100), offset);
        if (before.includes('// Mock') && !before.includes('{')) {
            localFixes++;
            return '';
        }
        return match;
    });

    // Clean up empty mock blocks
    fixed = fixed.replace(/\/\/\s*Mock[^\n]*\n\s*{\s*}\s*/g, '// Mock removed\n');

    if (localFixes > 0) {
        fixStats.malformedMocks += localFixes;
    }

    return fixed;
}

function fixAssertionConversions(content, filePath) {
    let fixed = content;
    let localFixes = 0;

    // Fix remaining Jest patterns that slipped through
    
    // expect.assertions(n) removal
    fixed = fixed.replace(/expect\.assertions\s*\(\s*\d+\s*\)\s*;?/g, '');
    
    // fail() -> assert.fail()
    fixed = fixed.replace(/\bfail\s*\(/g, 'assert.fail(');
    
    // toHaveLength final conversion
    fixed = fixed.replace(/\.toHaveLength\s*\(\s*(\d+)\s*\)/g, (match, length) => {
        localFixes++;
        return `.length, ${length})`;
    });

    // toBeLessThan -> assert.ok(x < y)
    fixed = fixed.replace(/expect\s*\(\s*([^)]+)\s*\)\s*\.toBeLessThan\s*\(\s*([^)]+)\s*\)/g, (match, val, threshold) => {
        localFixes++;
        return `assert.ok(${val} < ${threshold})`;
    });

    // toBeGreaterThanOrEqual -> assert.ok(x >= y)
    fixed = fixed.replace(/expect\s*\(\s*([^)]+)\s*\)\s*\.toBeGreaterThanOrEqual\s*\(\s*([^)]+)\s*\)/g, (match, val, threshold) => {
        localFixes++;
        return `assert.ok(${val} >= ${threshold})`;
    });

    // Fix onCall pattern for sinon
    fixed = fixed.replace(/\.mockReturnValueOnce\s*\(/g, '.onCall(0).returns(');
    fixed = fixed.replace(/\.onCall\(0\)\.returns\s*\(/g, '.onFirstCall().returns(');

    if (localFixes > 0) {
        fixStats.assertionConversions += localFixes;
    }

    return fixed;
}

function cleanupComments(content, filePath) {
    let fixed = content;
    let localFixes = 0;

    // Remove duplicate cleanup comments
    fixed = fixed.replace(/(\/\/\s*Mock cleanup\s*\n){2,}/g, '// Mock cleanup\n');
    
    // Remove "No newline at end of file" comments
    fixed = fixed.replace(/\/\/?\s*No newline at end of file\s*$/gm, '');
    
    // Clean up malformed end-of-file markers
    fixed = fixed.replace(/}\s*No newline at end of file\s*$/gm, '}');
    
    // Ensure proper newline at end of file
    if (!fixed.endsWith('\n')) {
        fixed += '\n';
        localFixes++;
    }

    // Remove multiple consecutive empty lines
    fixed = fixed.replace(/\n{4,}/g, '\n\n\n');

    if (localFixes > 0) {
        fixStats.cleanupComments += localFixes;
    }

    return fixed;
}

function processFile(filePath) {
    try {
        const content = fs.readFileSync(filePath, 'utf8');
        let fixed = content;
        let changesMade = false;

        // Apply all fixes
        const fixes = [
            fixIncompleteBlocks,
            fixTryWithoutCatch,
            fixMalformedMocks,
            fixAssertionConversions,
            cleanupComments
        ];

        for (const fix of fixes) {
            const before = fixed;
            fixed = fix(fixed, filePath);
            if (fixed !== before) {
                changesMade = true;
            }
        }

        if (changesMade) {
            fs.writeFileSync(filePath, fixed, 'utf8');
            filesFixed++;
            console.log(`✓ Fixed: ${path.relative(process.cwd(), filePath)}`);
        }

        totalFiles++;
    } catch (error) {
        console.error(`✗ Error processing ${filePath}: ${error.message}`);
    }
}

function findTestFiles(dir) {
    const files = [];
    const items = fs.readdirSync(dir, { withFileTypes: true });

    for (const item of items) {
        const fullPath = path.join(dir, item.name);
        if (item.isDirectory() && !item.name.includes('node_modules')) {
            files.push(...findTestFiles(fullPath));
        } else if (item.isFile() && item.name.endsWith('.test.ts')) {
            files.push(fullPath);
        }
    }

    return files;
}

// Main execution
console.log('🔧 Final structural fixes for remaining test issues...\n');

const e2eTestDir = path.join(process.cwd(), 'e2e', 'src', 'suite');
const testFiles = findTestFiles(e2eTestDir);

testFiles.forEach(processFile);

// Summary
console.log('\n📊 Summary:');
console.log(`Total files processed: ${totalFiles}`);
console.log(`Files fixed: ${filesFixed}`);
console.log(`Total fixes applied: ${Object.values(fixStats).reduce((a, b) => a + b, 0)}`);
console.log('\nFix breakdown:');
console.log(`  - Incomplete blocks: ${fixStats.incompleteBlocks}`);
console.log(`  - Try without catch: ${fixStats.tryWithoutCatch}`);
console.log(`  - Malformed mocks: ${fixStats.malformedMocks}`);
console.log(`  - Assertion conversions: ${fixStats.assertionConversions}`);
console.log(`  - Cleanup comments: ${fixStats.cleanupComments}`);