#!/usr/bin/env node

const fs = require('fs');
const { execSync } = require('child_process');

// Read the file list to process
const fileListPath = './files-to-fix.txt';
const fileList = fs.readFileSync(fileListPath, 'utf8')
  .split('\n')
  .filter(line => line.trim() && !line.startsWith('#'))
  .map(line => line.trim());

console.log(`🔧 Brute force fixing ${fileList.length} files based on actual TypeScript errors...`);

// Get error count
function getErrorCount() {
  try {
    const output = execSync('npx tsc --noEmit 2>&1 | grep "error TS" | wc -l', { 
      encoding: 'utf8',
      stdio: 'pipe'
    });
    return parseInt(output.trim());
  } catch (error) {
    return 9999;
  }
}

// Get errors for specific file
function getFileErrors(filePath) {
  try {
    const output = execSync(`npx tsc --noEmit 2>&1 | grep "${filePath}"`, { 
      encoding: 'utf8',
      stdio: 'pipe'
    });
    return output.split('\n').filter(line => line.includes('error TS')).length;
  } catch (error) {
    return 0;
  }
}

const initialErrorCount = getErrorCount();
console.log(`📊 Initial error count: ${initialErrorCount}`);

// Specific fixes based on the actual error patterns I see
const specificFixes = [
  // Fix provider-enablement-validation.test.ts missing brace after proxyquire
  {
    file: 'e2e/src/suite/api/provider-enablement-validation.test.ts',
    name: 'Fix missing closing brace after proxyquire',
    search: /const \{ buildApiHandler \} = proxyquire\('..\/..\/..\/..\/src\/api\/index', \{\s+\.\.\.services\/mcp\/integration\/McpIntegration': \{\s+McpIntegration: MockMcpIntegration\s+\}\s+\/\/ Mock cleanup\s+suite\("Provider Enablement Validation"/,
    replace: `const { buildApiHandler } = proxyquire('../../../../src/api/index', {
	'../services/mcp/integration/McpIntegration': {
		McpIntegration: MockMcpIntegration
	}
})

suite("Provider Enablement Validation"`
  },
  
  // Fix index.test.ts missing closing brace
  {
    file: 'e2e/src/suite/api/index.test.ts',
    name: 'Fix missing closing brace',
    search: /\/\/ Mock cleanup$/m,
    replace: '// Mock cleanup\n})'
  },
  
  // Fix ollama-integration.test.ts multiple issues
  {
    file: 'e2e/src/suite/api/ollama-integration.test.ts',
    name: 'Fix multiple catch block and structure issues',
    search: /\} catch \(_e: unknown\) \{\s+\/\/ Not valid JSON, treat as text\s+\} catch \(error\) \{\s+assert\.fail\("Unexpected error: " \+ error\.message\)\s+\}/,
    replace: `} catch (_e: unknown) {
						// Not valid JSON, treat as text
					}`
  }
];

let totalFilesFixed = 0;

// Apply specific fixes for each file
fileList.forEach(filePath => {
  if (!fs.existsSync(filePath)) {
    console.log(`⚠️  File not found: ${filePath}`);
    return;
  }

  console.log(`\n📁 Processing: ${filePath}`);
  
  const initialFileErrors = getFileErrors(filePath);
  console.log(`   📊 Initial errors: ${initialFileErrors}`);
  
  if (initialFileErrors === 0) {
    console.log(`   ✅ Already error-free, skipping`);
    return;
  }
  
  // Save original content
  const originalContent = fs.readFileSync(filePath, 'utf8');
  let content = originalContent;
  let fixesApplied = 0;
  
  // Apply specific fixes for this file
  specificFixes.forEach(fix => {
    if (fix.file === filePath) {
      const beforeContent = content;
      content = content.replace(fix.search, fix.replace);
      if (beforeContent !== content) {
        console.log(`   🔧 Applied: ${fix.name}`);
        fixesApplied++;
      }
    }
  });
  
  // Apply some generic patterns based on common TypeScript errors
  const genericFixes = [
    // Fix stray catch blocks without try
    {
      name: 'Fix stray catch blocks',
      search: /^(\s*)\} catch \(error\) \{\s*assert\.fail\(.*?\)\s*\}\s*$/gm,
      replace: ''
    },
    
    // Fix incomplete try blocks
    {
      name: 'Fix incomplete try-catch structure',
      search: /(\s*try \{[^}]*\n[^}]*)\n(\s*)\} catch \(error\) \{\s*assert\.fail\(.*?\)\s*\}/g,
      replace: '$1\n$2}'
    },
    
    // Fix missing commas in object literals
    {
      name: 'Fix missing commas',
      search: /(\w+: [^,\n}]+)\n(\s*)(\w+:)/g,
      replace: '$1,\n$2$3'
    },
    
    // Fix broken comment patterns that create syntax errors
    {
      name: 'Fix comment syntax breaks',
      search: /\/\/ Mock (\w+)[\s\n]*(\w)/g,
      replace: '// Mock $1\n$2'
    }
  ];
  
  // Apply generic fixes
  genericFixes.forEach(fix => {
    const beforeContent = content;
    content = content.replace(fix.search, fix.replace);
    if (beforeContent !== content) {
      console.log(`   🔧 Applied: ${fix.name}`);
      fixesApplied++;
    }
  });
  
  // Test if the fixes helped
  if (fixesApplied > 0) {
    fs.writeFileSync(filePath, content, 'utf8');
    const newFileErrors = getFileErrors(filePath);
    
    if (newFileErrors < initialFileErrors) {
      const reduction = initialFileErrors - newFileErrors;
      console.log(`   ✅ SUCCESS: Reduced ${reduction} errors (${initialFileErrors} → ${newFileErrors})`);
      totalFilesFixed++;
    } else if (newFileErrors === initialFileErrors) {
      console.log(`   📊 NEUTRAL: Applied fixes but no error change`);
    } else {
      console.log(`   ❌ WORSE: Increased errors, reverting`);
      fs.writeFileSync(filePath, originalContent, 'utf8');
    }
  } else {
    console.log(`   📋 No applicable fixes found`);
  }
});

const finalErrorCount = getErrorCount();
console.log(`\n🎯 SUMMARY:`);
console.log(`   📈 Files with improvements: ${totalFilesFixed}`);
console.log(`   📊 Overall: ${initialErrorCount} → ${finalErrorCount} errors`);

if (finalErrorCount < initialErrorCount) {
  console.log(`\n✅ Ready to commit ${initialErrorCount - finalErrorCount} error reduction!`);
} else {
  console.log(`\n📋 No overall improvement made`);
}