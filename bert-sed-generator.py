#!/usr/bin/env python3

import subprocess
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print("🧠 BERT Sed Pattern Generator")
print("Using BERT's existing knowledge to generate sed fixes!\n")

# Use a code-aware model
model_name = "microsoft/DialoGPT-medium"  # Good for generating commands
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def get_current_errors():
    """Get current TypeScript errors"""
    try:
        result = subprocess.run(['npx', 'tsc', '--noEmit'], 
                              capture_output=True, text=True, timeout=30)
        return result.stderr
    except:
        return ""

def generate_sed_pattern(error_context):
    """Use BERT to generate sed pattern for error"""
    
    prompt = f"""Given this TypeScript error context, generate a sed command to fix it:

{error_context}

Generate a sed pattern like: s/old_pattern/new_pattern/g

sed pattern:"""

    inputs = tokenizer.encode(prompt, return_tensors='pt', truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs, 
            max_length=inputs.shape[1] + 50,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract sed pattern from response
    if 's/' in response:
        sed_start = response.find('s/')
        sed_end = response.find('\n', sed_start)
        if sed_end == -1:
            sed_end = len(response)
        return response[sed_start:sed_end].strip()
    
    return None

def main():
    errors = get_current_errors()
    
    if not errors:
        print("✅ No TypeScript errors found!")
        return
    
    print(f"🔍 Found errors, generating BERT sed patterns...\n")
    
    # Get first few error lines
    error_lines = errors.split('\n')[:10]
    error_context = '\n'.join(error_lines)
    
    print(f"Error context:\n{error_context}\n")
    
    # Generate sed pattern
    pattern = generate_sed_pattern(error_context)
    
    if pattern:
        print(f"🎯 BERT generated sed pattern: {pattern}")
        
        # Write to file for sed-fix.js to use
        with open('bert-patterns.txt', 'w') as f:
            f.write(f"{pattern}\n")
        print("💾 Saved pattern to bert-patterns.txt")
        
    else:
        print("❌ Could not generate sed pattern")

if __name__ == "__main__":
    main()