#!/usr/bin/env node

const fs = require('fs');
const { execSync } = require('child_process');

// Read the file list to process
const fileListPath = './files-to-fix.txt';
const fileList = fs.readFileSync(fileListPath, 'utf8')
  .split('\n')
  .filter(line => line.trim() && !line.startsWith('#'))
  .map(line => line.trim());

console.log(`🔍 Processing ${fileList.length} files with validation-based fixes...`);

// Get initial error count
function getErrorCount() {
  try {
    const output = execSync('npx tsc --noEmit 2>&1 | grep "error TS" | wc -l', { 
      encoding: 'utf8',
      stdio: 'pipe'
    });
    return parseInt(output.trim());
  } catch (error) {
    console.log('⚠️  Could not get error count, assuming high number');
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
    return 0; // No errors found for this file
  }
}

// Define fix patterns for different error types
const fixSets = [
  {
    name: 'Basic try-catch repairs',
    fixes: [
      {
        search: /(\s+return \[{ matched: true, type: "reasoning", data: String\(jsonObj\.content\) }\s+)\} catch \(error\) \{\s+assert\.fail\('Unexpected error: ' \+ error\.message\)\s+\}\s+\]/g,
        replace: '$1}]'
      },
      {
        search: /(\s+for await \(const _chunk of stream\) \{\s+\/\/ Should throw before getting here\s+\}\s+)\} catch \(error\) \{\s+assert\.fail\('Unexpected error: ' \+ error\.message\)\s+\}\s+assert\.fail\("Should have thrown an error"\)/g,
        replace: '$1assert.fail("Should have thrown an error")'
      }
    ]
  },
  {
    name: 'Function call repairs',
    fixes: [
      {
        search: /buildApiHandler\({ apiProvider: "([^"]+)", apiKey: "test" }\s+\}\)\s+\} catch \(error\) \{\s+assert\.fail\('Unexpected error: ' \+ error\.message\)\s+\}\s+ as any\)/g,
        replace: 'buildApiHandler({ apiProvider: "$1", apiKey: "test" } as any)'
      },
      {
        search: /buildApiHandler\({ apiProvider: "([^"]+)" }\s+\}\)\s+\} catch \(error\) \{\s+assert\.fail\('Unexpected error: ' \+ error\.message\)\s+\}\s+ as any\)/g,
        replace: 'buildApiHandler({ apiProvider: "$1" } as any)'
      }
    ]
  },
  {
    name: 'Registry method repairs',
    fixes: [
      {
        search: /await registry\.getModels\("([^"]+)", \{\}\s+\}\)\s+\} catch \(error\) \{\s+assert\.fail\('Unexpected error: ' \+ error\.message\)\s+\}\s+\)/g,
        replace: 'await registry.getModels("$1", {})'
      }
    ]
  },
  {
    name: 'Loop structure repairs',
    fixes: [
      {
        search: /models\.forEach\(model => \{\s+((?:.*\n)*?)\s+\} catch \(error\) \{\s+assert\.fail\('Unexpected error: ' \+ error\.message\)\s+\}\s*\)/g,
        replace: 'models.forEach(model => {\n$1\n\t\t\t\t\t})'
      }
    ]
  },
  {
    name: 'Console log repairs',
    fixes: [
      {
        search: /console\.log\(`([^`]+) catch \(error\) \{\s+assert\.fail\('Unexpected error: ' \+ error\.message\)\s+\}([^`]*)`\)/g,
        replace: 'console.log(`$1$2`)'
      }
    ]
  }
];

const initialErrorCount = getErrorCount();
console.log(`📊 Initial error count: ${initialErrorCount}`);

let totalFilesImproved = 0;
let totalErrorsReduced = 0;

// Process each file individually
fileList.forEach((filePath, index) => {
  if (!fs.existsSync(filePath)) {
    console.log(`⚠️  File not found: ${filePath}`);
    return;
  }

  console.log(`\n📁 [${index + 1}/${fileList.length}] Processing: ${filePath}`);
  
  // Get initial error count for this file
  const initialFileErrors = getFileErrors(filePath);
  console.log(`   📊 Initial errors in file: ${initialFileErrors}`);
  
  // Save original content
  const originalContent = fs.readFileSync(filePath, 'utf8');
  let bestContent = originalContent;
  let bestErrorCount = initialFileErrors;
  let appliedFixes = [];

  // Try each fix set
  fixSets.forEach(fixSet => {
    console.log(`   🔧 Trying: ${fixSet.name}`);
    
    let testContent = originalContent;
    let fixesApplied = 0;
    
    // Apply all fixes in this set
    fixSet.fixes.forEach(fix => {
      const beforeContent = testContent;
      testContent = testContent.replace(fix.search, fix.replace);
      if (beforeContent !== testContent) {
        fixesApplied++;
      }
    });
    
    if (fixesApplied > 0) {
      // Write test content and check errors
      fs.writeFileSync(filePath, testContent, 'utf8');
      const newFileErrors = getFileErrors(filePath);
      
      console.log(`      Applied ${fixesApplied} fixes, errors: ${initialFileErrors} → ${newFileErrors}`);
      
      if (newFileErrors < bestErrorCount) {
        bestContent = testContent;
        bestErrorCount = newFileErrors;
        appliedFixes.push(fixSet.name);
        console.log(`      ✅ Improvement! Keeping these fixes.`);
      } else if (newFileErrors === bestErrorCount && fixesApplied > 0) {
        console.log(`      📊 No error change, but fixes applied cleanly.`);
      } else {
        console.log(`      ❌ No improvement or increased errors.`);
      }
    }
  });
  
  // Restore the best version
  fs.writeFileSync(filePath, bestContent, 'utf8');
  
  if (bestErrorCount < initialFileErrors) {
    const errorReduction = initialFileErrors - bestErrorCount;
    totalFilesImproved++;
    totalErrorsReduced += errorReduction;
    console.log(`   ✅ SUCCESS: Reduced ${errorReduction} errors in ${filePath}`);
    console.log(`      Applied fixes: ${appliedFixes.join(', ')}`);
  } else if (appliedFixes.length > 0) {
    console.log(`   📊 NEUTRAL: Applied fixes but no error count change`);
  } else {
    // Restore original if no improvements
    fs.writeFileSync(filePath, originalContent, 'utf8');
    console.log(`   📋 UNCHANGED: No beneficial fixes found`);
  }
});

console.log(`\n🎯 SUMMARY:`);
console.log(`   📈 Files improved: ${totalFilesImproved}/${fileList.length}`);
console.log(`   📉 Total errors reduced: ${totalErrorsReduced}`);

const finalErrorCount = getErrorCount();
console.log(`   📊 Overall: ${initialErrorCount} → ${finalErrorCount} errors`);

// Update the file list to remove files that now have zero errors
console.log(`\n🧹 Checking for files that are now error-free...`);
const remainingFiles = [];
const completedFiles = [];

fileList.forEach(filePath => {
  if (fs.existsSync(filePath)) {
    const fileErrors = getFileErrors(filePath);
    if (fileErrors === 0) {
      completedFiles.push(filePath);
      console.log(`   ✅ COMPLETED: ${filePath} (0 errors)`);
    } else {
      remainingFiles.push(filePath);
      console.log(`   📋 REMAINING: ${filePath} (${fileErrors} errors)`);
    }
  }
});

// Update the files-to-fix.txt list
if (completedFiles.length > 0) {
  console.log(`\n📝 Updating file list: removing ${completedFiles.length} completed files`);
  const updatedContent = remainingFiles.join('\n') + '\n';
  fs.writeFileSync(fileListPath, updatedContent, 'utf8');
  
  // Also create a completed files list for reference
  const completedContent = completedFiles.join('\n') + '\n';
  fs.writeFileSync('./files-completed.txt', completedContent, 'utf8');
  console.log(`   📄 Created files-completed.txt with ${completedFiles.length} completed files`);
}

if (finalErrorCount < initialErrorCount) {
  console.log(`\n✅ Ready to commit ${initialErrorCount - finalErrorCount} error reduction!`);
  console.log(`   📂 ${remainingFiles.length} files remaining for future fixes`);
} else {
  console.log(`\n📋 No overall improvement, but individual files may have been cleaned up.`);
  console.log(`   📂 ${remainingFiles.length} files remaining in list`);
}