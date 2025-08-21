#!/usr/bin/env node

const fs = require('fs');

// Read the file list to process
const fileListPath = './files-to-fix.txt';
const fileList = fs.readFileSync(fileListPath, 'utf8')
  .split('\n')
  .filter(line => line.trim() && !line.startsWith('#'))
  .map(line => line.trim());

console.log(`Processing ${fileList.length} files for manual syntax fixes...`);

let totalChanges = 0;

// Process each file individually with specific fixes
fileList.forEach(filePath => {
  if (!fs.existsSync(filePath)) {
    console.log(`⚠️  File not found: ${filePath}`);
    return;
  }

  console.log(`\n📁 Processing: ${filePath}`);
  
  let content = fs.readFileSync(filePath, 'utf8');
  let originalContent = content;
  
  if (filePath.includes('ollama-integration.test.ts')) {
    // Fix the broken return statement and catch blocks
    content = content.replace(
      /return \[{ matched: true, type: "reasoning", data: String\(jsonObj\.content\) } catch \(error\) \{\s+assert\.fail\('Unexpected error: ' \+ error\.message\)\s+\}\s*\]/g,
      'return [{ matched: true, type: "reasoning", data: String(jsonObj.content) }]'
    );
    
    // Fix the broken try-catch in error test
    content = content.replace(
      /for await \(const _chunk of stream\) \{\s+\/\/ Should throw before getting here\s+\} catch \(error\) \{\s+assert\.fail\('Unexpected error: ' \+ error\.message\)\s+\}assert\.fail\("Should have thrown an error"\)/g,
      'for await (const _chunk of stream) {\n\t\t\t\t// Should throw before getting here\n\t\t\t}\n\t\t\tassert.fail("Should have thrown an error")'
    );
  }
  
  if (filePath.includes('provider-integration-validation.test.ts')) {
    // Fix the broken buildApiHandler calls
    content = content.replace(
      /buildApiHandler\({ apiProvider: "([^"]+)", apiKey: "test" } catch \(error\) \{\s+assert\.fail\('Unexpected error: ' \+ error\.message\)\s+\}as any\)/g,
      'buildApiHandler({ apiProvider: "$1", apiKey: "test" } as any)'
    );
    
    content = content.replace(
      /buildApiHandler\({ apiProvider: "([^"]+)" } catch \(error\) \{\s+assert\.fail\('Unexpected error: ' \+ error\.message\)\s+\}as any\)/g,
      'buildApiHandler({ apiProvider: "$1" } as any)'
    );
    
    // Fix broken assert.ok pattern
    content = content.replace(
      /assert\.ok\(typeof handler\[method\] === "function",\s+`\$\{provider\} catch \(error\) \{\s+assert\.fail\('Unexpected error: ' \+ error\.message\)\s+\}should have \$\{method\} method`\s+\)/g,
      'assert.ok(typeof handler[method] === "function", `${provider} should have ${method} method`)'
    );
  }
  
  if (filePath.includes('all-dynamic-providers.runtime.test.ts')) {
    // Fix the broken forEach model validation
    content = content.replace(
      /models\.forEach\(model => \{\s+assert\.ok\(model\.hasOwnProperty\('id'\)\)\s+assert\.ok\(model\.hasOwnProperty\('name'\)\)\s+assert\.ok\(model\.hasOwnProperty\('capabilities'\)\)\s+assert\.strictEqual\(typeof model\.id, "string"\)\s+assert\.strictEqual\(typeof model\.name, "string"\)\s+expect\(Array\.isArray\(model\.capabilities\)\)\.toBe\(true\)\s+\} catch \(error\) \{\s+assert\.fail\('Unexpected error: ' \+ error\.message\)\s+\}\)/g,
      'models.forEach(model => {\n\t\t\t\t\t\tassert.ok(model.hasOwnProperty(\'id\'))\n\t\t\t\t\t\tassert.ok(model.hasOwnProperty(\'name\'))\n\t\t\t\t\t\tassert.ok(model.hasOwnProperty(\'capabilities\'))\n\t\t\t\t\t\tassert.strictEqual(typeof model.id, "string")\n\t\t\t\t\t\tassert.strictEqual(typeof model.name, "string")\n\t\t\t\t\t\texpect(Array.isArray(model.capabilities)).toBe(true)\n\t\t\t\t\t})'
    );
    
    // Fix the broken console.log pattern
    content = content.replace(
      /console\.log\(`✅ \$\{providerName\} catch \(error\) \{\s+assert\.fail\('Unexpected error: ' \+ error\.message\)\s+\}: Found \$\{models\.length\} static models`\)/g,
      'console.log(`✅ ${providerName}: Found ${models.length} static models`)'
    );
    
    // Fix the broken registry.getModels calls
    content = content.replace(
      /await registry\.getModels\("([^"]+)", \{\} catch \(error\) \{\s+assert\.fail\('Unexpected error: ' \+ error\.message\)\s+\}\)/g,
      'await registry.getModels("$1", {})'
    );
    
    // Fix the broken Promise.all mapping
    content = content.replace(
      /const models = await registry\.getModels\(provider, config\)\s+\/\/ Mock implementation removed\s+\/\/ \s+\}\s+\} catch \(error\) \{\s+assert\.fail\("Unexpected error: " \+ error\.message\)\s+\}/g,
      'const models = await registry.getModels(provider, config)\n\t\t\t\treturn { success: true, provider, modelCount: models.length }\n\t\t\t} catch (error) {\n\t\t\t\treturn { success: false, provider, error: error.message }\n\t\t\t}'
    );
  }
  
  if (filePath.includes('bedrock-custom-arn.test.ts')) {
    // Fix malformed try-catch patterns
    content = content.replace(
      /await handler\.completePrompt\("test"\)\s+assert\.fail\("Should have thrown an error for invalid ARN"\)\s+\} catch \(error\) \{\s+assert\.fail\('Unexpected error: ' \+ error\.message\)\s+\} catch \(error\) \{\s+assert\.ok\(error instanceof Error\)\s+assert\.ok\(error\.message\.includes\("Invalid ARN format"\)\)/g,
      'await handler.completePrompt("test")\n\t\t\tassert.fail("Should have thrown an error for invalid ARN")\n\t\t} catch (error) {\n\t\t\tassert.ok(error instanceof Error)\n\t\t\tassert.ok(error.message.includes("Invalid ARN format"))'
    );
    
    content = content.replace(
      /await handler\.completePrompt\("test"\)\s+assert\.fail\("Should have thrown an error for non-existent model"\)\s+\} catch \(error\) \{\s+assert\.fail\('Unexpected error: ' \+ error\.message\)\s+\} catch \(error\) \{\s+assert\.ok\(error instanceof Error\)\s+assert\.ok\(error\.message\.includes\("Model not found"\)\)/g,
      'await handler.completePrompt("test")\n\t\t\tassert.fail("Should have thrown an error for non-existent model")\n\t\t} catch (error) {\n\t\t\tassert.ok(error instanceof Error)\n\t\t\tassert.ok(error.message.includes("Model not found"))'
    );
  }
  
  if (filePath.includes('base-provider.schema-only.test.ts')) {
    // Fix the malformed tool handler try-catch
    content = content.replace(
      /tool\.handler\(\)\s+\} catch \(error\) \{\s+assert\.fail\('Unexpected error: ' \+ error\.message\)\s+\} catch \(error\) \{\s+if \(error\.message\.includes\("handled by MCP provider"\)\) \{\s+executionErrors\+\+\s+\}/g,
      'tool.handler()\n\t\t\t} catch (error) {\n\t\t\t\tif (error.message.includes("handled by MCP provider")) {\n\t\t\t\t\texecutionErrors++\n\t\t\t\t}'
    );
  }
  
  // Write back the fixed content
  if (content !== originalContent) {
    fs.writeFileSync(filePath, content, 'utf8');
    console.log(`   📝 Applied fixes to ${filePath}`);
    totalChanges++;
  } else {
    console.log(`   ✨ No changes needed for ${filePath}`);
  }
});

console.log(`\n🎯 Summary: Applied fixes to ${totalChanges} files`);
console.log(`\n🔍 Run TypeScript compiler to check error reduction...`);