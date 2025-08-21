#!/usr/bin/env node

const fs = require('fs');

// Read the file list to process
const fileListPath = './files-to-fix.txt';
const fileList = fs.readFileSync(fileListPath, 'utf8')
  .split('\n')
  .filter(line => line.trim() && !line.startsWith('#'))
  .map(line => line.trim());

console.log(`Processing ${fileList.length} files for precise syntax fixes...`);

const fixes = [
  // Fix broken try-catch pattern in ollama-integration.test.ts (lines 44-46, 228-230)
  {
    name: 'Fix broken catch blocks after return statements',
    search: /(\s+return \[{ matched: true, type: "reasoning", data: String\(jsonObj\.content\) }\s+)\} catch \(error\) \{\s+assert\.fail\('Unexpected error: ' \+ error\.message\)\s+\}\s+\]/g,
    replace: '$1}]'
  },

  // Fix malformed try blocks that end with missing closing structures
  {
    name: 'Fix missing closing in try-catch',
    search: /(\s+for await \(const _chunk of stream\) \{\s+\/\/ Should throw before getting here\s+\}\s+)\} catch \(error\) \{\s+assert\.fail\('Unexpected error: ' \+ error\.message\)\s+\}\s+assert\.fail\("Should have thrown an error"\)/g,
    replace: '$1assert.fail("Should have thrown an error")'
  },

  // Fix buildApiHandler calls that are broken
  {
    name: 'Fix broken buildApiHandler calls',
    search: /buildApiHandler\({ apiProvider: "([^"]+)", apiKey: "test" }\s+\}\)\s+\} catch \(error\) \{\s+assert\.fail\('Unexpected error: ' \+ error\.message\)\s+\}\s+ as any\)/g,
    replace: 'buildApiHandler({ apiProvider: "$1", apiKey: "test" } as any)'
  },

  // Fix registry.getModels calls that are broken
  {
    name: 'Fix broken registry.getModels calls',
    search: /await registry\.getModels\("([^"]+)", \{\}\s+\}\)\s+\} catch \(error\) \{\s+assert\.fail\('Unexpected error: ' \+ error\.message\)\s+\}\s+\)/g,
    replace: 'await registry.getModels("$1", {})'
  },

  // Fix forEach model validation that's broken
  {
    name: 'Fix broken forEach model validation',
    search: /(\s+models\.forEach\(model => \{\s+assert\.ok\(model\.hasOwnProperty\('id'\)\)\s+assert\.ok\(model\.hasOwnProperty\('name'\)\)\s+assert\.ok\(model\.hasOwnProperty\('capabilities'\)\)\s+assert\.strictEqual\(typeof model\.id, "string"\)\s+assert\.strictEqual\(typeof model\.name, "string"\)\s+expect\(Array\.isArray\(model\.capabilities\)\)\.toBe\(true\)\s+\}\s+)\} catch \(error\) \{\s+assert\.fail\('Unexpected error: ' \+ error\.message\)\s+\}\s+\)/g,
    replace: '$1})'
  },

  // Fix console.log statements that are broken
  {
    name: 'Fix broken console.log statements',
    search: /console\.log\(`✅ \$\{providerName\}\s+\}\)\s+\} catch \(error\) \{\s+assert\.fail\('Unexpected error: ' \+ error\.message\)\s+\}\s+: Found \$\{models\.length\} static models`\)/g,
    replace: 'console.log(`✅ ${providerName}: Found ${models.length} static models`)'
  },

  // Fix assert statements that got mangled
  {
    name: 'Fix broken assert statements',
    search: /assert\.ok\(typeof handler\[method\] === "function",\s+`\$\{provider\}\s+\}\)\s+\} catch \(error\) \{\s+assert\.fail\('Unexpected error: ' \+ error\.message\)\s+\}\s+should have \$\{method\} method`\s+\)/g,
    replace: 'assert.ok(typeof handler[method] === "function", `${provider} should have ${method} method`)'
  },

  // Fix incomplete Promise.all mapping
  {
    name: 'Fix broken Promise.all mapping',
    search: /(\s+const models = await registry\.getModels\(provider, config\)\s+\/\/ Mock implementation removed\s+\/\/ \s+\}\s+)\} catch \(error\) \{\s+assert\.fail\('Unexpected error: ' \+ error\.message\)\s+\}/g,
    replace: '$1return { success: true, provider, modelCount: models.length }\n\t\t\t} catch (error) {\n\t\t\t\treturn { success: false, provider, error: error.message }\n\t\t\t}'
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
    const originalContent = content;
    content = content.replace(fix.search, fix.replace);
    
    if (originalContent !== content) {
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