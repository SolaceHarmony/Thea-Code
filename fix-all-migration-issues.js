#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const glob = require('glob');

function fixTestFile(filePath) {
    let content = fs.readFileSync(filePath, 'utf8');
    let changed = false;
    
    // Fix remaining Jest references
    if (content.includes('jest.')) {
        content = content.replace(/jest\.SpyInstance/g, 'sinon.SinonSpy');
        content = content.replace(/jest\.MockedFunction<[^>]+>/g, 'sinon.SinonStub');
        content = content.replace(/jest\.Mocked<[^>]+>/g, 'any');
        content = content.replace(/jest\.clearAllMocks\(\)/g, 'sinon.restore()');
        content = content.replace(/jest\.resetAllMocks\(\)/g, 'sinon.restore()');
        content = content.replace(/jest\.spyOn\(([^,]+),\s*['"]([^'"]+)['"]\)/g, 'sinon.spy($1, "$2")');
        content = content.replace(/jest\.requireMock/g, '// TODO: requireMock');
        content = content.replace(/jest\.requireActual/g, '// TODO: requireActual');
        content = content.replace(/\(globalThis as any\)\.jest/g, '// TODO: global jest reference');
        changed = true;
    }
    
    // Fix expect patterns that weren't converted
    if (content.includes('expect(')) {
        // More expect patterns
        content = content.replace(/expect\(([^)]+)\)\.toBeGreaterThan\(([^)]+)\)/g, 'assert.ok($1 > $2)');
        content = content.replace(/expect\(([^)]+)\)\.toBeLessThan\(([^)]+)\)/g, 'assert.ok($1 < $2)');
        content = content.replace(/expect\(([^)]+)\)\.toHaveBeenCalledWith\(([^)]+)\)/g, 'assert.ok($1.calledWith($2))');
        content = content.replace(/expect\(([^)]+)\)\.not\.toHaveBeenCalled\(\)/g, 'assert.ok(!$1.called)');
        content = content.replace(/expect\(([^)]+)\)\.toHaveBeenCalled\(\)/g, 'assert.ok($1.called)');
        content = content.replace(/expect\(([^)]+)\)\.resolves\.toEqual\(([^)]+)\)/g, 'assert.deepStrictEqual(await $1, $2)');
        content = content.replace(/expect\(([^)]+)\)\.rejects\.toThrow\(([^)]+)\)/g, 'await assert.rejects($1, $2)');
        content = content.replace(/await expect\(([^)]+)\)\.rejects\.toThrow\(([^)]+)\)/g, 'await assert.rejects(async () => $1, $2)');
        
        // Fix expect.any patterns
        content = content.replace(/expect\.any\(Object\)/g, 'sinon.match.object');
        content = content.replace(/expect\.any\(String\)/g, 'sinon.match.string');
        content = content.replace(/expect\.any\(Number\)/g, 'sinon.match.number');
        content = content.replace(/expect\.any\(Boolean\)/g, 'sinon.match.bool');
        content = content.replace(/expect\.any\(Function\)/g, 'sinon.match.func');
        content = content.replace(/expect\.any\(Array\)/g, 'sinon.match.array');
        
        changed = true;
    }
    
    // Fix mock return patterns
    if (content.includes('.mock')) {
        content = content.replace(/\.mockResolvedValueOnce\(/g, '.resolvesOnce(');
        content = content.replace(/\.mockRejectedValueOnce\(/g, '.rejectsOnce(');
        content = content.replace(/\.mockReturnValueOnce\(/g, '.returnsOnce(');
        content = content.replace(/\.mockRestore\(\)/g, '.restore()');
        content = content.replace(/\.mock\.calls/g, '.args');
        content = content.replace(/\.mockImplementationOnce\(/g, '.onFirstCall().callsFake(');
        changed = true;
    }
    
    // Fix sinon.SinonStubed typo
    if (content.includes('sinon.SinonStubed')) {
        content = content.replace(/sinon\.SinonStubed/g, 'sinon.SinonStubStatic');
        changed = true;
    }
    
    // Fix .called patterns
    if (content.includes('.toHaveBeenCalled')) {
        content = content.replace(/expect\(([^)]+)\)\.not\.toHaveBeenCalled\(\)/g, 'assert.ok(!$1.called)');
        content = content.replace(/expect\(([^)]+)\)\.toHaveBeenCalled\(\)/g, 'assert.ok($1.called)');
        content = content.replace(/expect\(([^)]+)\)\.toHaveBeenCalledTimes\(([^)]+)\)/g, 'assert.strictEqual($1.callCount, $2)');
        changed = true;
    }
    
    // Fix console spy patterns
    if (content.includes("sinon.spy(console,")) {
        content = content.replace(/sinon\.spy\(console,\s*'error'\)\.mockImplementation\(\)/g, "sinon.stub(console, 'error')");
        content = content.replace(/sinon\.spy\(console,\s*'warn'\)\.mockImplementation\(\)/g, "sinon.stub(console, 'warn')");
        content = content.replace(/sinon\.spy\(console,\s*'log'\)\.mockImplementation\(\)/g, "sinon.stub(console, 'log')");
        changed = true;
    }
    
    // Remove "No newline at end of file" mid-file
    if (content.includes('No newline at end of file') && !content.endsWith('No newline at end of file')) {
        content = content.replace(/\n\s*No newline at end of file\s*\n/g, '\n');
        changed = true;
    }
    
    // Ensure proper newline at end
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
const testFiles = glob.sync('src/e2e/src/suite/**/*.e2e.test.ts');
console.log(`Processing ${testFiles.length} test files...`);

let fixedCount = 0;
for (const file of testFiles) {
    if (fixTestFile(file)) {
        console.log(`Fixed: ${path.basename(file)}`);
        fixedCount++;
    }
}

console.log(`\nFixed ${fixedCount} files`);