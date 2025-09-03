#!/usr/bin/env node

const fs = require('fs');

// Only target files with 1 error (safest to fix)
const singleErrorFiles = [
  'e2e/src/suite/api/providers/anthropic.test.ts',
  'e2e/src/suite/api/providers/anthropic.edge-cases.test.ts', 
  'e2e/src/suite/api/providers.test.ts',
  'e2e/src/suite/api/providers-comprehensive.test.ts',
  'e2e/src/suite/api/index.test.ts'
];

console.log('🎯 Applying minimal safe fixes to single-error files...');

singleErrorFiles.forEach(filePath => {
  if (!fs.existsSync(filePath)) {
    console.log(`⚠️  File not found: ${filePath}`);
    return;
  }

  console.log(`\n📁 Processing: ${filePath}`);
  let content = fs.readFileSync(filePath, 'utf8');
  const originalContent = content;
  
  // Very simple and safe fix: Fix trailing "// Mock cleanup" comments that break syntax
  if (content.endsWith('// Mock cleanup') && !content.endsWith('// Mock cleanup\n')) {
    content = content.replace(/\/\/ Mock cleanup$/, '// Mock cleanup\n');
    console.log('   🔧 Fixed trailing comment');
  }
  
  // Fix broken "// Mock cleanu" + "p" pattern seen in some files
  content = content.replace(/\/\/ Mock cleanu\s*\n\s*p\s*$/m, '// Mock cleanup');
  
  if (content !== originalContent) {
    fs.writeFileSync(filePath, content, 'utf8');
    console.log(`   ✅ Applied fix to ${filePath}`);
  } else {
    console.log(`   📋 No changes needed for ${filePath}`);
  }
});

console.log('\n🔍 Safe minimal fixes complete');