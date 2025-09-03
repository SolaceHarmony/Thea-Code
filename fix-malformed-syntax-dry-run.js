#!/usr/bin/env node

/**
 * Surgical fix for malformed syntax in test files
 * This script identifies and reports common patterns of malformed syntax
 * from the Jest to Mocha migration
 */

const fs = require('fs');
const path = require('path');
const glob = require('glob');

// Common malformed patterns we've seen
const PATTERNS = {
    // Malformed comment blocks
    malformedCommentImport: {
        pattern: /^import\s+.*?\s+from\s+.*?\/\*\*[\s\S]*?\*\//gm,
        description: 'Import statement followed by comment block',
        example: 'import * as sinon from \'sinon\'/**\n * Test suite\n */'
    },
    
    // Mixed up import lines
    malformedImportAfterBrace: {
        pattern: /^import\s+\{[\s\S]*?^import\s+\*\s+as/gm,
        description: 'Import statement inside another import',
        example: 'import {\nimport * as assert from \'assert\''
    },
    
    // Broken TODO comments
    malformedTodoComment: {
        pattern: /\/\/\s*TODO:\s*Use proxyquire.*?\s*-\s*"[^"]+",\s*\(\)\s*=>\s*\{/g,
        description: 'Malformed TODO comment for proxyquire',
        example: '// TODO: Use proxyquire for module mocking - "@aws-sdk/client-bedrock-runtime", () => ({'
    },
    
    // Incomplete closing brackets
    unclosedProxyquire: {
        pattern: /proxyquire\([^)]+\)\)[^)]*$/gm,
        description: 'Proxyquire call with mismatched parentheses',
        example: 'BedrockRuntimeClient: sinon.stub(() => mockClient),\n}))' 
    },
    
    // Mock patterns mixed with code
    inlineMockAssignments: {
        pattern: /^[\t ]*}[\t ]*\)\)[\t ]*$/gm,
        description: 'Floating closing brackets from incomplete mocking',
        example: '        }))'
    },
    
    // Test syntax issues
    testEachSyntax: {
        pattern: /\.each\s*\(/g,
        description: 'Jest .each syntax that needs conversion',
        example: 'test.each(['
    },
    
    // Expect syntax
    expectSyntax: {
        pattern: /expect\(/g,
        description: 'Jest expect calls that need conversion to assert',
        count: 0
    },
    
    // Mock implementation issues
    mockImplementationOnce: {
        pattern: /\.mockImplementationOnce\(/g,
        description: 'Jest mock methods needing conversion to sinon',
        count: 0
    },
    
    // Improper suite/test structure
    standaloneSetup: {
        pattern: /^setup\(/gm,
        description: 'setup() outside of suite block',
        count: 0
    },
    
    // Misplaced imports
    importsAfterCode: {
        pattern: /^[^i\n][^m\n][^p\n][\s\S]*?^import\s+/gm,
        description: 'Import statements after code has started',
        count: 0
    }
};

// Proposed fixes
const FIXES = {
    malformedCommentImport: (content) => {
        // Fix: import * as sinon from 'sinon'/** comment */ -> proper separation
        return content.replace(
            /^(import\s+.*?\s+from\s+['""][^'"]+['"])(\/\*\*[\s\S]*?\*\/)/gm,
            '$1\n\n$2'
        );
    },
    
    malformedImportAfterBrace: (content) => {
        // Fix: move malformed imports out of other imports
        return content.replace(
            /^(import\s+\{[^\}]*)\n(import\s+\*\s+as\s+\w+\s+from\s+['""][^'"]+['"])/gm,
            '$2\n$1'
        );
    },
    
    malformedTodoComment: (content) => {
        // Fix TODO comments to be proper comments without code
        return content.replace(
            /\/\/\s*TODO:\s*Use proxyquire.*?\s*-\s*("[^"]+"),\s*\(\)\s*=>\s*\{/g,
            '// TODO: Use proxyquire for module mocking - $1\n// Mock implementation needed here'
        );
    },
    
    testEachSyntax: (content) => {
        // Convert .each syntax to forEach
        const lines = content.split('\n');
        let inEach = false;
        let result = [];
        
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            if (line.includes('.each(')) {
                // Start of .each block - needs manual review
                result.push('// TODO: Convert .each to forEach pattern');
                result.push(line.replace('.each(', '.forEach(// NEEDS MANUAL CONVERSION: '));
                inEach = true;
            } else {
                result.push(line);
            }
        }
        return result.join('\n');
    },
    
    expectSyntax: (content) => {
        // Basic expect to assert conversions
        return content
            .replace(/expect\((.*?)\)\.toBe\((.*?)\)/g, 'assert.strictEqual($1, $2)')
            .replace(/expect\((.*?)\)\.toEqual\((.*?)\)/g, 'assert.deepStrictEqual($1, $2)')
            .replace(/expect\((.*?)\)\.toBeTruthy\(\)/g, 'assert.ok($1)')
            .replace(/expect\((.*?)\)\.toBeFalsy\(\)/g, 'assert.ok(!$1)')
            .replace(/expect\((.*?)\)\.toBeUndefined\(\)/g, 'assert.strictEqual($1, undefined)')
            .replace(/expect\((.*?)\)\.toBeNull\(\)/g, 'assert.strictEqual($1, null)')
            .replace(/expect\((.*?)\)\.toContain\((.*?)\)/g, 'assert.ok($1.includes($2))')
            .replace(/expect\((.*?)\)\.toHaveLength\((.*?)\)/g, 'assert.strictEqual($1.length, $2)')
            .replace(/expect\((.*?)\)\.toThrow\(\)/g, 'assert.throws(() => $1)')
            .replace(/expect\((.*?)\)\.not\./g, 'assert.ok(!/* NEEDS REVIEW: not. */ $1.');
    },
    
    mockImplementationOnce: (content) => {
        // Convert Jest mocks to sinon
        return content.replace(
            /(\w+)\.mockImplementationOnce\(/g,
            '$1.onFirstCall().callsFake('
        );
    }
};

function analyzeFile(filePath) {
    const content = fs.readFileSync(filePath, 'utf8');
    const issues = [];
    
    for (const [key, pattern] of Object.entries(PATTERNS)) {
        const matches = content.match(pattern.pattern);
        if (matches && matches.length > 0) {
            issues.push({
                type: key,
                description: pattern.description,
                count: matches.length,
                examples: matches.slice(0, 2)
            });
        }
    }
    
    return issues;
}

function proposeFixForFile(filePath) {
    let content = fs.readFileSync(filePath, 'utf8');
    const originalContent = content;
    const appliedFixes = [];
    
    // Apply fixes in order
    for (const [fixType, fixFn] of Object.entries(FIXES)) {
        const before = content;
        content = fixFn(content);
        if (before !== content) {
            appliedFixes.push(fixType);
        }
    }
    
    return {
        changed: originalContent !== content,
        appliedFixes,
        content
    };
}

// Main execution
console.log('🔍 Analyzing test files for malformed syntax patterns...\n');

const testFiles = glob.sync('e2e/src/suite/**/*.test.ts');
const analysisResults = {};
const fixProposals = {};

// Analyze all files
for (const file of testFiles) {
    const issues = analyzeFile(file);
    if (issues.length > 0) {
        analysisResults[file] = issues;
        const proposal = proposeFixForFile(file);
        if (proposal.changed) {
            fixProposals[file] = proposal;
        }
    }
}

// Report findings
console.log('📊 ANALYSIS RESULTS\n');
console.log('='.repeat(80));

const totalFiles = Object.keys(analysisResults).length;
console.log(`Found issues in ${totalFiles} files:\n`);

// Group by issue type
const issueTypes = {};
for (const [file, issues] of Object.entries(analysisResults)) {
    for (const issue of issues) {
        if (!issueTypes[issue.type]) {
            issueTypes[issue.type] = {
                files: [],
                totalCount: 0,
                description: issue.description
            };
        }
        issueTypes[issue.type].files.push(file);
        issueTypes[issue.type].totalCount += issue.count;
    }
}

// Report by issue type
for (const [type, data] of Object.entries(issueTypes)) {
    console.log(`\n${type}:`);
    console.log(`  Description: ${data.description}`);
    console.log(`  Found in ${data.files.length} files (${data.totalCount} occurrences)`);
    console.log(`  Sample files:`);
    data.files.slice(0, 3).forEach(f => {
        console.log(`    - ${f.replace(/^.*\/e2e\//, 'e2e/')}`);
    });
}

// Report fix proposals
console.log('\n\n🔧 FIX PROPOSALS\n');
console.log('='.repeat(80));

const totalFixable = Object.keys(fixProposals).length;
console.log(`Can automatically fix ${totalFixable} files:\n`);

// Show sample fixes
let samplesShown = 0;
for (const [file, proposal] of Object.entries(fixProposals)) {
    if (samplesShown >= 5) break;
    
    console.log(`\n${file.replace(/^.*\/e2e\//, 'e2e/')}:`);
    console.log(`  Applied fixes: ${proposal.appliedFixes.join(', ')}`);
    samplesShown++;
}

// Summary
console.log('\n\n📋 SUMMARY\n');
console.log('='.repeat(80));
console.log(`Total test files scanned: ${testFiles.length}`);
console.log(`Files with issues: ${totalFiles}`);
console.log(`Files with automatic fixes available: ${totalFixable}`);

// High-priority issues
const highPriority = [
    'malformedCommentImport',
    'malformedImportAfterBrace', 
    'malformedTodoComment',
    'expectSyntax'
];

console.log('\n🚨 High-priority patterns to fix:');
for (const pattern of highPriority) {
    if (issueTypes[pattern]) {
        console.log(`  - ${pattern}: ${issueTypes[pattern].files.length} files`);
    }
}

// Save detailed report
const report = {
    timestamp: new Date().toISOString(),
    summary: {
        totalFiles: testFiles.length,
        filesWithIssues: totalFiles,
        filesFixable: totalFixable
    },
    issueTypes,
    analysisResults,
    fixProposals: Object.keys(fixProposals)
};

fs.writeFileSync(
    'malformed-syntax-analysis.json',
    JSON.stringify(report, null, 2)
);

console.log('\n✅ Detailed report saved to malformed-syntax-analysis.json');
console.log('\n📝 To apply fixes, run: node fix-malformed-syntax-dry-run.js --apply');

// Check if we should apply fixes
if (process.argv.includes('--apply')) {
    console.log('\n\n🚀 APPLYING FIXES...\n');
    
    for (const [file, proposal] of Object.entries(fixProposals)) {
        try {
            fs.writeFileSync(file, proposal.content);
            console.log(`✅ Fixed: ${file.replace(/^.*\/e2e\//, 'e2e/')}`);
        } catch (error) {
            console.log(`❌ Error fixing ${file}: ${error.message}`);
        }
    }
    
    console.log('\n✅ Fixes applied! Run TypeScript compiler to check results.');
} else {
    console.log('\n💡 This was a dry run. No files were modified.');
}