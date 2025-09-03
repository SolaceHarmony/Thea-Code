#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

// Read the file list to process
const fileListPath = './files-to-fix.txt';
const fileList = fs.readFileSync(fileListPath, 'utf8')
  .split('\n')
  .filter(line => line.trim() && !line.startsWith('#'))
  .map(line => line.trim());

console.log(`Processing ${fileList.length} files for TypeScript error fixes...`);

// Define the search and replace patterns for common syntax errors
const fixes = [
  // Fix malformed try-catch blocks - case 1: } catch (error) { without proper try
  {
    name: 'Fix dangling catch blocks',
    search: /(\s+)\} catch \(error\) \{\s*assert\.fail\('Unexpected error: ' \+ error\.message\)\s*\}\s*(\w+|$)/g,
    replace: '$1} catch (error) {\n$1\tassert.fail(\'Unexpected error: \' + error.message)\n$1}\n$1$2'
  },

  // Fix incomplete try blocks - case 2: try { without catch
  {
    name: 'Fix incomplete try blocks',
    search: /(.*try \{[^}]*\}) catch \(error\) \{\s*assert\.fail\('Unexpected error: ' \+ error\.message\)\s*\}([^}]*)/g,
    replace: '$1\n\t\t} catch (error) {\n\t\t\tassert.fail(\'Unexpected error: \' + error.message)\n\t\t}\n$2'
  },

  // Fix multiple catch blocks pattern
  {
    name: 'Fix multiple catch pattern',
    search: /\} catch \(error\) \{\s*assert\.fail\('Unexpected error: ' \+ error\.message\)\s*\}\s*\} catch \(error\) \{\s*assert\.fail\("Unexpected error: " \+ error\.message\)\s*\}/g,
    replace: '\t\t} catch (error) {\n\t\t\tassert.fail(\'Unexpected error: \' + error.message)\n\t\t}'
  },

  // Fix missing closing braces in conditionals
  {
    name: 'Fix missing closing braces',
    search: /(\s+if \([^)]+\) \{[^}]+)\n(\s+\} else \{)/g,
    replace: '$1\n$2'
  },

  // Fix incomplete object literals
  {
    name: 'Fix incomplete object structures',
    search: /(\w+): \{\s*\} catch \(error\) \{\s*assert\.fail\('Unexpected error: ' \+ error\.message\)\s*\}/g,
    replace: '$1: {}\n\t\t} catch (error) {\n\t\t\tassert.fail(\'Unexpected error: \' + error.message)\n\t\t}'
  },

  // Fix malformed async function calls
  {
    name: 'Fix malformed async calls',
    search: /await registry\.getModels\([^)]+\) catch \(error\) \{\s*assert\.fail\('Unexpected error: ' \+ error\.message\)\s*\}/g,
    replace: 'await registry.getModels(providerName, config)\n\t\t\t} catch (error) {\n\t\t\t\tassert.fail(\'Unexpected error: \' + error.message)\n\t\t\t}'
  },

  // Fix mixed quote patterns in error messages
  {
    name: 'Standardize error message quotes',
    search: /assert\.fail\("Unexpected error: " \+ error\.message\)/g,
    replace: 'assert.fail(\'Unexpected error: \' + error.message)'
  },

  // Fix malformed comment patterns that break syntax
  {
    name: 'Fix malformed comment patterns',
    search: /\/\/ Mock (\w+)[\s\n]*\} catch \(error\) \{\s*assert\.fail\('Unexpected error: ' \+ error\.message\)\s*\}/g,
    replace: '// Mock $1'
  },

  // Fix incomplete array/object closing
  {
    name: 'Fix incomplete array/object patterns',
    search: /(\s+\])(\s+\} catch \(error\) \{\s*assert\.fail\('Unexpected error: ' \+ error\.message\)\s*\})/g,
    replace: '$1'
  },

  // Fix stray catch blocks after comments
  {
    name: 'Remove stray catch blocks after comments',
    search: /\/\/ Mock cleanup[\s\n]*\} catch \(error\) \{\s*assert\.fail\('Unexpected error: ' \+ error\.message\)\s*\}/g,
    replace: '// Mock cleanup'
  },

  // Fix incomplete method calls
  {
    name: 'Fix incomplete method calls',
    search: /([a-zA-Z_$][a-zA-Z0-9_$]*\([^)]*) catch \(error\) \{\s*assert\.fail\('Unexpected error: ' \+ error\.message\)\s*\}/g,
    replace: '$1)\n\t\t} catch (error) {\n\t\t\tassert.fail(\'Unexpected error: \' + error.message)\n\t\t}'
  }
];

let totalChanges = 0;

// Process each file from the list
fileList.forEach(filePath => {
  if (!fs.existsSync(filePath)) {
    console.log(`⚠️  File not found: ${filePath}`);
    return;
  }

  console.log(`\n📁 Processing: ${filePath}`);
  
  let content = fs.readFileSync(filePath, 'utf8');
  let fileChanges = 0;
  
  // Apply each fix pattern
  fixes.forEach(fix => {
    const beforeLength = content.length;
    content = content.replace(fix.search, fix.replace);
    const afterLength = content.length;
    
    if (beforeLength !== afterLength) {
      const matches = (content.match(fix.search) || []).length;
      console.log(`   ✅ ${fix.name}: Applied`);
      fileChanges++;
    }
  });
  
  // Write back the fixed content
  if (fileChanges > 0) {
    fs.writeFileSync(filePath, content, 'utf8');
    console.log(`   📝 Applied ${fileChanges} fixes to ${filePath}`);
    totalChanges += fileChanges;
  } else {
    console.log(`   ✨ No changes needed for ${filePath}`);
  }
});

console.log(`\n🎯 Summary: Applied ${totalChanges} total fixes across ${fileList.length} files`);
console.log(`\n🔍 Run TypeScript compiler to check error reduction...`);