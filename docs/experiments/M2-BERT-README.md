# M2-BERT: Monarch Mixer BERT for TypeScript/ESLint Error Correction

A sophisticated code correction system using Stanford's Monarch Mixer (M2) architecture with Qwen3-Coder as teacher model.

## Architecture

- **Student Model**: M2-BERT with subquadratic O(n log n) attention
- **Teacher Model**: Qwen3-Coder:30B (via Ollama)
- **Training**: Multiple strategies including ESLint-driven, chaos engineering, and self-supervised
- **Output**: Unified diffs for code fixes (not full file rewrites)

## Components

### Training Components

- **train-eslint**: ESLint-driven training with raw errors and tight context
- **train-chaos**: Corrupt code systematically and learn to repair
- **train-self**: Self-supervised learning from linter feedback
- **train-continuous**: Continuous learning with memory replay
- **train-semantic**: Semantic validation to prevent degenerate fixes

### Operational Components

- **generate-diffs**: Generate unified diffs for TypeScript fixes
- **validate**: Symbiotic validation with linter and Qwen3
- **test**: Comprehensive smoke test suite
- **evaluate**: MAD (Mechanistic Architecture Design) evaluation

## Quick Start

```bash
# Run smoke tests
python3 m2-bert.py test --quick

# List all components
python3 m2-bert.py list

# Train with ESLint errors
python3 m2-bert.py train-eslint --max-errors 50

# Generate diffs for current errors
python3 m2-bert.py generate-diffs --cycles 3

# Run chaos training
python3 m2-bert.py train-chaos --sessions 5
```

## Component Testing

```bash
# Test specific component
python3 m2-bert-smoke-test.py --component eslint --iterations 1

# List available components
python3 m2-bert-smoke-test.py --list-components

# Run full test suite
python3 m2-bert-smoke-test.py --verbose
```

## Key Features

1. **ESLint Integration**: Direct training from ESLint errors with grep-like context
2. **Teacher-Student Learning**: Qwen3-Coder teaches M2-BERT to fix code correctly
3. **Semantic Validation**: Prevents degenerate solutions (commenting out, @ts-ignore)
4. **Diff Generation**: Produces minimal diffs instead of full file rewrites
5. **Continuous Learning**: JSON-based memory with replay training
6. **Chaos Engineering**: Systematic code corruption for robustness

## Architecture Details

### Monarch Mixer (M2)
- Subquadratic attention mechanism from Stanford's Hazy Research
- O(n log n) complexity instead of O(n²)
- Efficient for long sequences

### Training Pipeline
1. Extract errors (TypeScript/ESLint)
2. Get tight context (grep -A/-B style)
3. Generate fix with M2-BERT
4. Validate with teacher (Qwen3)
5. Check semantic validity
6. Apply if approved

### Safety Features
- Semantic validation prevents degenerate fixes
- Linter validation ensures syntactic correctness
- Teacher model provides ground truth
- Diff-only output prevents catastrophic changes

## Requirements

- Python 3.8+
- PyTorch (with MPS support for M1/M2 Macs)
- Transformers library
- Ollama with Qwen3-Coder model
- Node.js with TypeScript and ESLint

## Installation

```bash
# Install Python dependencies
pip install torch transformers requests

# Install Ollama and pull Qwen3-Coder
ollama pull qwen3-coder:30b

# Install Node dependencies
npm install -g typescript eslint
```

## Training Data Format

The system uses JSON for training data persistence:

```json
{
  "error": "error TS2304: Cannot find name 'console'",
  "file": "src/index.ts",
  "line": 42,
  "context": "const x = 5;\nconsole.log(x);",
  "correct_diff": "@@ -1,2 +1,2 @@\n const x = 5;\n-console.log(x);\n+console.log(x);",
  "timestamp": "2024-01-20T10:30:00Z"
}
```

## Performance

- Training: ~100 errors per minute on M1/M2 Mac
- Inference: <1 second per fix
- Memory: ~4GB for M2-BERT, additional for Qwen3 via Ollama
- Success rate: 70-85% syntactic, 60-75% semantic (after training)

## Credits

Based on research from:
- Stanford Hazy Research (Monarch Mixer, Hyena, MAD)
- Microsoft (CodeBERT base model)
- Alibaba (Qwen3-Coder teacher model)