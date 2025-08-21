#!/usr/bin/env node

const fs = require('fs');
const { execSync } = require('child_process');

console.log('🧠 Error Pattern Analyzer - Using Cosine Similarity for Fix Discovery\n');

// Get all TypeScript errors
function getAllErrors() {
  try {
    const output = execSync('npx tsc --noEmit 2>&1 | grep "error TS"', { 
      encoding: 'utf8',
      stdio: 'pipe'
    });
    return output.split('\n').filter(line => line.trim());
  } catch (error) {
    return [];
  }
}

// Extract context around error using grep
function getErrorContext(filename, lineNumber, contextLines = 3) {
  try {
    const output = execSync(`grep -n -A${contextLines} -B${contextLines} "." "${filename}" | grep -A${contextLines} -B${contextLines} ":${lineNumber}:"`, { 
      encoding: 'utf8',
      stdio: 'pipe'
    });
    return output;
  } catch (error) {
    // Fallback: read the file directly
    try {
      const content = fs.readFileSync(filename, 'utf8').split('\n');
      const start = Math.max(0, lineNumber - contextLines - 1);
      const end = Math.min(content.length, lineNumber + contextLines);
      return content.slice(start, end).map((line, i) => `${start + i + 1}:${line}`).join('\n');
    } catch (e) {
      return '';
    }
  }
}

// Convert text to vector for cosine similarity
function textToVector(text) {
  // Simple tokenization and frequency counting
  const tokens = text.toLowerCase()
    .replace(/[^\w\s]/g, ' ')  // Replace punctuation with spaces
    .split(/\s+/)
    .filter(token => token.length > 0);
  
  const freq = {};
  tokens.forEach(token => {
    freq[token] = (freq[token] || 0) + 1;
  });
  
  return freq;
}

// Calculate cosine similarity between two vectors
function cosineSimilarity(vecA, vecB) {
  const allKeys = new Set([...Object.keys(vecA), ...Object.keys(vecB)]);
  
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;
  
  for (const key of allKeys) {
    const a = vecA[key] || 0;
    const b = vecB[key] || 0;
    
    dotProduct += a * b;
    normA += a * a;
    normB += b * b;
  }
  
  if (normA === 0 || normB === 0) return 0;
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

// Parse error line to extract file, line number, and error type
function parseError(errorLine) {
  const match = errorLine.match(/^(.+?)\((\d+),(\d+)\): error (TS\d+): (.+)$/);
  if (!match) return null;
  
  return {
    file: match[1],
    line: parseInt(match[2]),
    column: parseInt(match[3]),
    errorCode: match[4],
    message: match[5],
    fullLine: errorLine
  };
}

// Main analysis
console.log('📊 Collecting all TypeScript errors...');
const errors = getAllErrors();
console.log(`Found ${errors.length} errors\n`);

if (errors.length === 0) {
  console.log('🎉 No errors found!');
  process.exit(0);
}

console.log('🔍 Extracting context for each error...');
const errorData = [];

for (let i = 0; i < errors.length; i++) {
  const error = parseError(errors[i]);
  if (!error) continue;
  
  console.log(`   [${i + 1}/${errors.length}] ${error.file}:${error.line} (${error.errorCode})`);
  
  const context = getErrorContext(error.file, error.line);
  const vector = textToVector(context);
  
  errorData.push({
    ...error,
    context,
    vector,
    id: i
  });
}

console.log(`\n🧮 Calculating cosine similarities for ${errorData.length} error contexts...\n`);

// Find clusters using cosine similarity
const clusters = [];
const processed = new Set();
const SIMILARITY_THRESHOLD = 0.3; // Adjust this threshold

for (let i = 0; i < errorData.length; i++) {
  if (processed.has(i)) continue;
  
  const cluster = [errorData[i]];
  processed.add(i);
  
  for (let j = i + 1; j < errorData.length; j++) {
    if (processed.has(j)) continue;
    
    const similarity = cosineSimilarity(errorData[i].vector, errorData[j].vector);
    
    if (similarity > SIMILARITY_THRESHOLD) {
      cluster.push(errorData[j]);
      processed.add(j);
    }
  }
  
  clusters.push({
    size: cluster.size,
    errors: cluster,
    avgSimilarity: cluster.length > 1 ? 
      cluster.reduce((sum, err1, idx1) => 
        sum + cluster.reduce((innerSum, err2, idx2) => 
          idx1 !== idx2 ? innerSum + cosineSimilarity(err1.vector, err2.vector) : innerSum, 0), 0) / 
      (cluster.length * (cluster.length - 1)) : 1
  });
}

// Sort clusters by size (largest first)
clusters.sort((a, b) => b.size - a.size);

console.log('🎯 ERROR PATTERN CLUSTERS:\n');

clusters.forEach((cluster, i) => {
  if (cluster.size === 1) return; // Skip singleton clusters
  
  console.log(`📦 CLUSTER ${i + 1}: ${cluster.size} similar errors (avg similarity: ${cluster.avgSimilarity.toFixed(3)})`);
  
  // Show error types in this cluster
  const errorTypes = {};
  const files = new Set();
  
  cluster.errors.forEach(error => {
    errorTypes[error.errorCode] = (errorTypes[error.errorCode] || 0) + 1;
    files.add(error.file);
  });
  
  console.log(`   Error types: ${Object.entries(errorTypes).map(([code, count]) => `${code}(${count})`).join(', ')}`);
  console.log(`   Files: ${files.size} files`);
  
  // Show a sample context
  console.log(`   Sample context:`);
  const sampleContext = cluster.errors[0].context.split('\n').slice(0, 5).join('\n');
  console.log(`   ${sampleContext.replace(/\n/g, '\n   ')}`);
  
  // Generate potential fix pattern
  const commonTokens = findCommonTokens(cluster.errors.map(e => e.context));
  if (commonTokens.length > 0) {
    console.log(`   🔧 Common patterns: ${commonTokens.slice(0, 5).join(', ')}`);
  }
  
  console.log('');
});

// Show singleton patterns (unique errors)
const singletons = clusters.filter(c => c.size === 1);
if (singletons.length > 0) {
  console.log(`🔹 UNIQUE ERRORS: ${singletons.length} errors with no similar patterns\n`);
  
  singletons.slice(0, 5).forEach(cluster => {
    const error = cluster.errors[0];
    console.log(`   ${error.errorCode}: ${error.file}:${error.line} - ${error.message}`);
  });
  
  if (singletons.length > 5) {
    console.log(`   ... and ${singletons.length - 5} more unique errors`);
  }
}

// Generate fix suggestions
console.log('\n🚀 SUGGESTED FIX PATTERNS:\n');

clusters.slice(0, 3).forEach((cluster, i) => {
  if (cluster.size === 1) return;
  
  console.log(`💡 Pattern ${i + 1}: Fix ${cluster.size} similar errors`);
  
  const errorCodes = Object.keys(cluster.errors.reduce((acc, e) => ({ ...acc, [e.errorCode]: true }), {}));
  const files = [...new Set(cluster.errors.map(e => e.file))];
  
  console.log(`   Target: ${errorCodes.join(', ')} errors in ${files.length} files`);
  
  // Simple pattern detection for fix suggestions
  const contexts = cluster.errors.map(e => e.context);
  if (contexts.some(c => c.includes('} catch') && c.includes('assert.fail'))) {
    console.log(`   🔧 Suggested fix: Remove extra catch blocks`);
    console.log(`   📝 Command: node sed-fix.js "} catch.*assert.fail.*}" "" ${files.slice(0, 3).join(' ')}`);
  } else if (contexts.some(c => c.includes('// Mock cleanup') && !c.includes('})'))) {
    console.log(`   🔧 Suggested fix: Add missing closing braces`);
    console.log(`   📝 Command: node sed-fix.js "// Mock cleanup$" "// Mock cleanup\\n})" ${files.slice(0, 3).join(' ')}`);
  } else {
    console.log(`   🔧 Manual review recommended - patterns detected but no automatic fix available`);
  }
  
  console.log('');
});

function findCommonTokens(contexts) {
  const allTokens = contexts.map(context => 
    context.toLowerCase().split(/\s+/).filter(t => t.length > 2)
  );
  
  const tokenCounts = {};
  allTokens.forEach(tokens => {
    const unique = [...new Set(tokens)];
    unique.forEach(token => {
      tokenCounts[token] = (tokenCounts[token] || 0) + 1;
    });
  });
  
  return Object.entries(tokenCounts)
    .filter(([token, count]) => count >= Math.ceil(contexts.length * 0.5)) // Present in at least 50% of contexts
    .sort((a, b) => b[1] - a[1])
    .map(([token]) => token);
}