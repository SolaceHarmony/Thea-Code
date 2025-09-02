#/usr/bin/env python3
"""
M2-BERT Diff Validator
Only allows diffs that pass linting AND get Qwen3's approval
Like a code-cleaning symbiotic relationship
"""

import subprocess
import tempfile
import os
import requests
from typing import Dict, Tuple, Optional

class DiffValidator:
    """
    Validates M2-BERT's diffs before allowing them through
    """
    
    def __init__(self):
        self.approved_diffs = []
        self.rejected_diffs = []
        
    def apply_diff_to_temp(self, file_path: str, diff: str) -> Optional[str]:
        """
        Apply diff to a temporary copy of the file
        Returns path to temp file if successful
        """
        try:
            # Create temp file with original content
            with open(file_path, 'r') as f:
                original_content = f.read()
            
            with tempfile.NamedTemporaryFile(mode = 'w', suffix = '.ts', delete = False) as tmp:
                tmp.write(original_content)
                temp_path = tmp.name
            
            # Try to apply the diff using patch command
            with tempfile.NamedTemporaryFile(mode = 'w', suffix = '.diff', delete = False) as diff_file:
                diff_file.write(diff)
                diff_path = diff_file.name
            
            # Apply patch
            result = subprocess.run(
                ['patch', temp_path, diff_path],
                capture_output = True,
                text = True,
                timeout = 5
            )
            
            # Clean up diff file
            os.unlink(diff_path)
            
            if result.returncode == 0:
                return temp_path
            else:
                os.unlink(temp_path)
                return None
                
        except Exception as e:
            print(f"Error applying diff: {e}")
            return None
    
    def run_linter(self, file_path: str) -> Tuple[bool, str]:
        """
        Run TypeScript compiler on file to check for errors
        Returns (success, output)
        """
        try:
            result = subprocess.run(
                ['npx', 'tsc', '--noEmit', '--skipLibCheck', file_path],
                capture_output = True,
                text = True,
                timeout = 10
            )
            
            # Check if clean (no errors)
            has_errors = 'error TS' in result.stderr or 'error TS' in result.stdout
            
            return not has_errors, result.stderr + result.stdout
            
        except Exception as e:
            return False, str(e)
    
    def ask_qwen_approval(self, original_code: str, fixed_code: str, diff: str) -> Tuple[bool, str]:
        """
        Ask Qwen3 if the fix looks good
        Returns (approved, feedback)
        """
        prompt = f"""Review this code fix. Is it correct and safe?
Respond with JSON: {{"approved": true/false, "reason": "brief explanation"}}

ORIGINAL CODE:
{original_code[:500]}

DIFF APPLIED:
{diff[:300]}

RESULTING CODE:
{fixed_code[:500]}

JSON Response:"""
        
        try:
            response = requests.post('http://localhost:11434/api/generate',
                json = {
                    'model': 'qwen3-coder:30b',
                    'prompt': prompt,
                    'stream': False,
                    'options': {
                        'temperature': 0.1,
                    }
                }, timeout = 20)
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get('response', '')
                
                # Try to parse JSON response
                import json
                try:
                    # Extract JSON from response
                    if '{' in response_text:
                        json_start = response_text.index('{')
                        json_end = response_text.rindex('}') + 1
                        json_str = response_text[json_start:json_end]
                        
                        approval = json.loads(json_str)
                        return approval.get('approved', False), approval.get('reason', 'No reason given')
                except:
                    pass
                
                # Fallback: look for keywords
                response_lower = response_text.lower()
                if 'approved' in response_lower or 'looks good' in response_lower or 'correct' in response_lower:
                    return True, "Approved"
                elif 'not approved' in response_lower or 'incorrect' in response_lower or 'wrong' in response_lower:
                    return False, "Not approved"
            
        except Exception as e:
            print(f"Qwen approval error: {e}")
        
        return False, "Could not get approval"
    
    def validate_diff(self, file_path: str, diff: str, error_info: Dict) -> Dict:
        """
        Full validation pipeline for a diff
        Returns validation result with details
        """
        result = {
            'diff': diff[:200],  # Store truncated for logging
            'file': file_path,
            'error': error_info.get('error', ''),
            'lint_passed': False,
            'qwen_approved': False,
            'approved': False,
            'feedback': []
        }
        
        print(f"\n Validating diff for {file_path}.")
        
        # Step 1: Apply diff to temp file
        temp_path = self.apply_diff_to_temp(file_path, diff)
        
        if not temp_path:
            result['feedback'].append("Failed to apply diff")
            print("   Diff could not be applied")
            self.rejected_diffs.append(result)
            return result
        
        print("   Diff applied to temp file")
        
        # Step 2: Run linter
        lint_success, lint_output = self.run_linter(temp_path)
        result['lint_passed'] = lint_success
        
        if not lint_success:
            result['feedback'].append(f"Linter failed: {lint_output[:100]}")
            print(f"   Linter check failed")
            os.unlink(temp_path)
            self.rejected_diffs.append(result)
            return result
        
        print("   Linter check passed")
        
        # Step 3: Get Qwen approval
        with open(file_path, 'r') as f:
            original = f.read()
        with open(temp_path, 'r') as f:
            fixed = f.read()
        
        qwen_approved, qwen_reason = self.ask_qwen_approval(original, fixed, diff)
        result['qwen_approved'] = qwen_approved
        result['feedback'].append(f"Qwen: {qwen_reason}")
        
        if not qwen_approved:
            print(f"   Qwen rejected: {qwen_reason}")
            os.unlink(temp_path)
            self.rejected_diffs.append(result)
            return result
        
        print(f"   Qwen approved: {qwen_reason}")
        
        # Clean up temp file
        os.unlink(temp_path)
        
        # All checks passed
        result['approved'] = True
        self.approved_diffs.append(result)
        
        print("   DIFF APPROVED AND READY TO APPLY")
        
        return result
    
    def apply_approved_diff(self, file_path: str, diff: str) -> bool:
        """
        Actually apply an approved diff to the real file
        """
        try:
            # Save the diff to a file
            with tempfile.NamedTemporaryFile(mode = 'w', suffix = '.diff', delete = False) as f:
                f.write(diff)
                diff_path = f.name
            
            # Apply the patch
            result = subprocess.run(
                ['patch', file_path, diff_path],
                capture_output = True,
                text = True,
                timeout = 5
            )
            
            # Clean up
            os.unlink(diff_path)
            
            if result.returncode == 0:
                print(f"   Diff successfully applied to {file_path}")
                return True
            else:
                print(f"   Failed to apply diff: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"   Error applying diff: {e}")
            return False
    
    def print_summary(self):
        """
        Print validation summary
        """
        print("\n" + " = "*60)
        print("VALIDATION SUMMARY")
        print(" = "*60)
        print(f" Approved diffs: {len(self.approved_diffs)}")
        print(f" Rejected diffs: {len(self.rejected_diffs)}")
        
        if self.rejected_diffs:
            print("\nRejection reasons:")
            for diff in self.rejected_diffs[-5:]:  # Show last 5
                print(f"  - {diff['file']}: {', '.join(diff['feedback'][:1])}")
        
        if self.approved_diffs:
            print("\nApproved fixes:")
            for diff in self.approved_diffs[-5:]:  # Show last 5
                print(f"   {diff['file']}: {diff['error'][:50]}")

class SymbioticFixer:
    """
    The complete symbiotic system:
    M2-BERT generates, Validator checks, only clean code gets through
    """
    
    def __init__(self):
        self.validator = DiffValidator()
        
    def fix_with_validation(self, m2bert_diff: str, file_path: str, error_info: Dict) -> bool:
        """
        Try to fix with M2-BERT's diff, but only if validated
        """
        # Validate the diff
        validation = self.validator.validate_diff(file_path, m2bert_diff, error_info)
        
        if validation['approved']:
            # Apply the approved diff
            success = self.validator.apply_approved_diff(file_path, m2bert_diff)
            return success
        else:
            print(f"  Diff rejected: {', '.join(validation['feedback'])}")
            return False
    
    def run_symbiotic_loop(self):
        """
        Run the full symbiotic cleaning loop
        Like birds cleaning alligator teeth - mutual benefit
        """
        print("\n SYMBIOTIC CODE CLEANING")
        print(" = "*60)
        print("M2-BERT generates diffs")
        print("Linter + Qwen3 validate")
        print("Only clean code gets through")
        print(" = "*60)
        
        # This would integrate with the M2-BERT diff generator
        # For demo, showing the validation flow
        
        print("\nThe symbiotic relationship:")
        print("   M2-BERT: 'I made a diff'")
        print("   Linter: 'Let me check that.'")
        print("  ‍ Qwen3: 'Is this good code?'")
        print("   System: 'All clear, applying'")
        print("\nJust like cleaner birds and alligators")
        print("Everyone benefits, code stays healthy ")

if __name__ == "__main__":
    fixer = SymbioticFixer()
    fixer.run_symbiotic_loop()
    
    # In production, this would be called from m2-bert-diff-generator.py
    # after each diff is generated