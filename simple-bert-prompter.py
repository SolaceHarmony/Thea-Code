#!/usr/bin/env python3

import subprocess
from transformers import pipeline
import re

print("🧠 Simple BERT Sed Pattern Generator")

# Use a text generation model
generator = pipeline('text-generation', model='microsoft/DialoGPT-small')

def get_sample_errors():
    """Get a sample of current errors"""
    try:
        result = subprocess.run(['npx', 'tsc', '--noEmit'], 
                              capture_output=True, text=True, timeout=30)
        
        # Get first 5 error lines
        error_lines = [line for line in result.stderr.split('\n') 
                      if 'error TS' in line][:5]
        return error_lines
    except:
        return []

def prompt_bert_for_sed(error_sample):
    """Prompt BERT to suggest sed patterns"""
    
    prompt = f"""TypeScript errors need sed fixes:

{chr(10).join(error_sample)}

Common patterns:
- Remove imports: s/import.*from.*;//g  
- Fix expect: s/expect(.*).toBe(.*)/assert.strictEqual($1, $2)/g
- Remove assert.fail: s/assert.fail(.*)/\/\/ Test assertion removed/g
- Fix catch blocks: s/}} catch/}} catch (error) {{/g

Generate 3 sed patterns to fix these errors:"""

    try:
        response = generator(prompt, max_length=len(prompt) + 200, num_return_sequences=1)
        return response[0]['generated_text']
    except:
        return "Error generating patterns"

def main():
    errors = get_sample_errors()
    
    if not errors:
        print("✅ No errors to analyze")
        return
        
    print(f"🔍 Analyzing {len(errors)} error samples...\n")
    
    for i, error in enumerate(errors[:3]):
        print(f"Error {i+1}: {error}")
    
    print("\n🧠 Prompting BERT for sed patterns...\n")
    
    result = prompt_bert_for_sed(errors)
    print("BERT Response:")
    print("=" * 50)
    print(result)
    
    # Extract any sed patterns from response
    sed_patterns = re.findall(r's/[^/]+/[^/]*/g', result)
    
    if sed_patterns:
        print(f"\n🎯 Extracted {len(sed_patterns)} sed patterns:")
        for pattern in sed_patterns:
            print(f"  {pattern}")
            
        # Save patterns
        with open('bert-sed-patterns.txt', 'w') as f:
            for pattern in sed_patterns:
                f.write(f"{pattern}\n")
        print("\n💾 Saved to bert-sed-patterns.txt")
    else:
        print("\n❌ No sed patterns found in response")

if __name__ == "__main__":
    main()