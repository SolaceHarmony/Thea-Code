#!/usr/bin/env node
/**
 * Script to fix remaining malformed syntax in test files
 * Targets specific patterns identified in TypeScript errors
 */

const fs = require('fs');
const path = require('path');

let totalFiles = 0;
let filesFixed = 0;
let totalFixes = 0;

// Track specific fix types
const fixStats = {
    assertDeepStrictEqual: 0,
    malformedImports: 0,
    expectToAssert: 0,
    todoComments: 0,
    proxyquireMocks: 0,
    bracketIssues: 0
};

function fixAssertDeepStrictEqual(content, filePath) {
    let fixed = content;
    let localFixes = 0;

    // Fix assert.deepStrictEqual with TODO comments and malformed objects
    // Pattern: assert.deepStrictEqual(requestBody, \n\t\t\t\t// TODO: Object partial match - {\n
    const malformedAssertPattern = /assert\.deepStrictEqual\s*\(\s*([^,]+),\s*\n\s*\/\/\s*TODO:[^\n]+\s*-?\s*{\n/gm;
    if (malformedAssertPattern.test(fixed)) {
        fixed = fixed.replace(malformedAssertPattern, (match, arg1) => {
            localFixes++;
            return `assert.deepStrictEqual(${arg1}, {\n`;
        });
    }

    // Fix closing parentheses that are doubled
    // Pattern: }), \n\t\t\t)
    const doubleClosePattern = /}\s*\)\s*,\s*\n\s*\)/gm;
    if (doubleClosePattern.test(fixed)) {
        fixed = fixed.replace(doubleClosePattern, '})');
        localFixes++;
    }

    if (localFixes > 0) {
        fixStats.assertDeepStrictEqual += localFixes;
    }

    return fixed;
}

function fixMalformedImports(content, filePath) {
    let fixed = content;
    let localFixes = 0;

    // Fix broken import statements with "import { " pattern
    // Pattern: import type { \nimport { ApiHandlerOptions
    const brokenImportPattern = /^import\s+type\s*{\s*\nimport\s*{([^}]+)}/gm;
    if (brokenImportPattern.test(fixed)) {
        fixed = fixed.replace(brokenImportPattern, 'import type { $1 }');
        localFixes++;
    }

    // Fix imports split across lines incorrectly
    // Pattern: } from "../../../shared/neutral-history"\nimport { XmlMatcher
    const splitImportPattern = /}\s*from\s*["'][^"']+["']\s*\nimport\s*{/gm;
    if (splitImportPattern.test(fixed)) {
        fixed = fixed.replace(splitImportPattern, (match) => {
            localFixes++;
            return match.replace(/\nimport\s*{/, '\nimport {');
        });
    }

    if (localFixes > 0) {
        fixStats.malformedImports += localFixes;
    }

    return fixed;
}

function fixExpectPatterns(content, filePath) {
    let fixed = content;
    let localFixes = 0;

    // Convert expect().toHaveProperty() to assert checks
    const expectHavePropertyPattern = /expect\s*\(\s*([^)]+)\s*\)\s*\.toHaveProperty\s*\(\s*["']([^"']+)["']\s*\)/g;
    if (expectHavePropertyPattern.test(fixed)) {
        fixed = fixed.replace(expectHavePropertyPattern, (match, obj, prop) => {
            localFixes++;
            return `assert.ok(${obj}.hasOwnProperty('${prop}'))`;
        });
    }

    // Convert expect().not.toHaveProperty() to assert checks
    const expectNotHavePropertyPattern = /expect\s*\(\s*([^)]+)\s*\)\s*\.not\.toHaveProperty\s*\(\s*["']([^"']+)["']\s*\)/g;
    if (expectNotHavePropertyPattern.test(fixed)) {
        fixed = fixed.replace(expectNotHavePropertyPattern, (match, obj, prop) => {
            localFixes++;
            return `assert.ok(!${obj}.hasOwnProperty('${prop}'))`;
        });
    }

    // Convert expect().toContain() to assert.ok with includes
    const expectContainPattern = /expect\s*\(\s*([^)]+)\s*\)\s*\.toContain\s*\(\s*([^)]+)\s*\)/g;
    if (expectContainPattern.test(fixed)) {
        fixed = fixed.replace(expectContainPattern, (match, arr, item) => {
            localFixes++;
            return `assert.ok(${arr}.includes(${item}))`;
        });
    }

    // Convert expect().not.toContain() to assert.ok with !includes
    const expectNotContainPattern = /expect\s*\(\s*([^)]+)\s*\)\s*\.not\.toContain\s*\(\s*([^)]+)\s*\)/g;
    if (expectNotContainPattern.test(fixed)) {
        fixed = fixed.replace(expectNotContainPattern, (match, arr, item) => {
            localFixes++;
            return `assert.ok(!${arr}.includes(${item}))`;
        });
    }

    // Convert expect.any(Error) to sinon.match.instanceOf(Error)
    const expectAnyErrorPattern = /expect\.any\s*\(\s*Error\s*\)/g;
    if (expectAnyErrorPattern.test(fixed)) {
        fixed = fixed.replace(expectAnyErrorPattern, 'sinon.match.instanceOf(Error)');
        localFixes++;
    }

    // Convert expect(async).resolves.not.toThrow() to proper assert pattern
    const expectResolvesNotThrowPattern = /await\s+expect\s*\(\s*([^)]+)\s*\)\s*\.resolves\.not\.toThrow\s*\(\s*\)/g;
    if (expectResolvesNotThrowPattern.test(fixed)) {
        fixed = fixed.replace(expectResolvesNotThrowPattern, (match, expr) => {
            localFixes++;
            return `await assert.doesNotReject(async () => ${expr})`;
        });
    }

    // Convert expect(async).rejects.toThrow() to assert.rejects
    const expectRejectsThrowPattern = /await\s+expect\s*\(\s*([^)]+)\s*\)\s*\.rejects\.toThrow\s*\(\s*([^)]*)\s*\)/g;
    if (expectRejectsThrowPattern.test(fixed)) {
        fixed = fixed.replace(expectRejectsThrowPattern, (match, expr, errorMsg) => {
            localFixes++;
            if (errorMsg && errorMsg.trim()) {
                return `await assert.rejects(async () => ${expr}, new Error(${errorMsg}))`;
            }
            return `await assert.rejects(async () => ${expr})`;
        });
    }

    if (localFixes > 0) {
        fixStats.expectToAssert += localFixes;
    }

    return fixed;
}

function fixProxyquireMocks(content, filePath) {
    let fixed = content;
    let localFixes = 0;

    // Fix malformed mock TODO comments that break the syntax
    // Pattern: // TODO: Use proxyquire for module mocking - "modulename", () => {
    const malformedMockPattern = /\/\/\s*TODO:\s*Use proxyquire[^-]+-\s*["']([^"']+)["']\s*,\s*\(\)\s*=>\s*{\s*\n/gm;
    if (malformedMockPattern.test(fixed)) {
        fixed = fixed.replace(malformedMockPattern, (match, moduleName) => {
            localFixes++;
            return `// TODO: Mock setup needs manual migration for "${moduleName}"\n// `;
        });
    }

    // Fix dangling mock blocks
    // Pattern: }))  (extra closing parentheses from broken mocks)
    const danglingMockClosePattern = /^}\)\)$/gm;
    if (danglingMockClosePattern.test(fixed)) {
        fixed = fixed.replace(danglingMockClosePattern, '// Mock cleanup needed');
        localFixes++;
    }

    if (localFixes > 0) {
        fixStats.proxyquireMocks += localFixes;
    }

    return fixed;
}

function fixBracketIssues(content, filePath) {
    let fixed = content;
    let localFixes = 0;

    // Fix mismatched brackets in suite/test blocks
    let openBrackets = 0;
    let closeBrackets = 0;
    const lines = fixed.split('\n');
    
    lines.forEach(line => {
        openBrackets += (line.match(/\{/g) || []).length;
        closeBrackets += (line.match(/\}/g) || []).length;
    });

    // Add missing closing brackets at the end if needed
    const bracketDiff = openBrackets - closeBrackets;
    if (bracketDiff > 0 && bracketDiff <= 3) {
        fixed += '\n' + '})'.repeat(bracketDiff);
        localFixes += bracketDiff;
        fixStats.bracketIssues += bracketDiff;
    }

    return fixed;
}

function processFile(filePath) {
    try {
        const content = fs.readFileSync(filePath, 'utf8');
        let fixed = content;
        let changesMade = false;

        // Apply fixes in sequence
        const beforeFix = fixed;
        
        fixed = fixAssertDeepStrictEqual(fixed, filePath);
        if (fixed !== beforeFix) changesMade = true;
        
        const afterAssert = fixed;
        fixed = fixMalformedImports(fixed, filePath);
        if (fixed !== afterAssert) changesMade = true;
        
        const afterImports = fixed;
        fixed = fixExpectPatterns(fixed, filePath);
        if (fixed !== afterImports) changesMade = true;
        
        const afterExpect = fixed;
        fixed = fixProxyquireMocks(fixed, filePath);
        if (fixed !== afterExpect) changesMade = true;
        
        const afterMocks = fixed;
        fixed = fixBracketIssues(fixed, filePath);
        if (fixed !== afterMocks) changesMade = true;

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
console.log('🔧 Fixing remaining malformed syntax in test files...\n');

const e2eTestDir = path.join(process.cwd(), 'e2e', 'src', 'suite');
const testFiles = findTestFiles(e2eTestDir);

// Focus on files with the most errors first
const priorityFiles = [
    'api/providers/openai-native.test.ts',
    'api/providers/openai.edge-cases.test.ts',
    'api/providers/openai-usage-tracking.test.ts',
    'api/providers/bedrock.edge-cases.test.ts',
    'services/mcp/'
];

// Sort files to process priority files first
testFiles.sort((a, b) => {
    const aPriority = priorityFiles.some(p => a.includes(p)) ? 0 : 1;
    const bPriority = priorityFiles.some(p => b.includes(p)) ? 0 : 1;
    return aPriority - bPriority;
});

testFiles.forEach(processFile);

// Summary
console.log('\n📊 Summary:');
console.log(`Total files processed: ${totalFiles}`);
console.log(`Files fixed: ${filesFixed}`);
console.log(`Total fixes applied: ${Object.values(fixStats).reduce((a, b) => a + b, 0)}`);
console.log('\nFix breakdown:');
console.log(`  - assert.deepStrictEqual fixes: ${fixStats.assertDeepStrictEqual}`);
console.log(`  - Malformed imports: ${fixStats.malformedImports}`);
console.log(`  - expect() to assert conversions: ${fixStats.expectToAssert}`);
console.log(`  - TODO comment fixes: ${fixStats.todoComments}`);
console.log(`  - Proxyquire mock fixes: ${fixStats.proxyquireMocks}`);
console.log(`  - Bracket issues: ${fixStats.bracketIssues}`);