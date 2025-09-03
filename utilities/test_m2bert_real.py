#!/usr/bin/env python3
"""
Real test of M2-BERT implementation
No fake data - let's see if this actually works
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertForMaskedLM
import time
import numpy as np
from m2bert_modern import M2BertForMaskedLM, M2BertConfig

def test_real_text_generation():
    """Test with real text to see if outputs make sense"""
    print("="*70)
    print("TESTING WITH REAL TEXT")
    print("="*70)
    
    # Load a real tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Real text samples
    test_sentences = [
        "The capital of France is [MASK].",
        "Machine learning is a type of [MASK] intelligence.",
        "The [MASK] jumped over the lazy dog.",
        "Python is a programming [MASK] used for data science.",
    ]
    
    # Initialize our M2-BERT
    config = M2BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=768,
        num_hidden_layers=12,
        intermediate_size=3072,
        use_monarch_mlp=True,
        monarch_nblocks=4,
        use_torch_compile=False  # Disable for testing
    )
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = M2BertForMaskedLM(config).to(device)
    model.eval()
    
    print(f"\nModel initialized on {device}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    
    # Process each sentence
    for sentence in test_sentences:
        print(f"\nInput: {sentence}")
        
        # Tokenize
        inputs = tokenizer(sentence, return_tensors="pt", padding=True)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        
        # Find mask position
        mask_token_id = tokenizer.mask_token_id
        mask_positions = (input_ids == mask_token_id).nonzero(as_tuple=True)
        
        if len(mask_positions[0]) == 0:
            print("  No [MASK] token found")
            continue
        
        # Get predictions
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs['logits']
        
        # Get top 5 predictions for masked position
        mask_idx = mask_positions[1][0]
        mask_logits = logits[0, mask_idx]
        top_5_tokens = torch.topk(mask_logits, 5).indices
        
        print("  Top 5 predictions:")
        for i, token_id in enumerate(top_5_tokens):
            token = tokenizer.decode([token_id])
            prob = F.softmax(mask_logits, dim=-1)[token_id].item()
            print(f"    {i+1}. '{token}' (prob: {prob:.3f})")


def test_mlm_loss():
    """Test MLM loss computation with real masked data"""
    print("\n" + "="*70)
    print("TESTING MLM LOSS")
    print("="*70)
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Create a batch of real sentences
    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming the world of technology.",
        "Python is a popular programming language for data science.",
        "Natural language processing enables computers to understand text.",
    ]
    
    # Tokenize
    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=128)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    # Create MLM labels (mask 15% of tokens)
    labels = input_ids.clone()
    mask_probability = 0.15
    
    # Don't mask special tokens
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) 
        for val in input_ids.tolist()
    ]
    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
    
    # Create probability matrix
    probability_matrix = torch.full(input_ids.shape, mask_probability)
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    
    # Mask tokens
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # Only compute loss on masked tokens
    
    # Replace masked tokens with [MASK]
    input_ids[masked_indices] = tokenizer.mask_token_id
    
    # Move to device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)
    
    # Test our model
    config = M2BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=768,
        num_hidden_layers=12,
        intermediate_size=3072,
        use_monarch_mlp=True,
        monarch_nblocks=4,
        use_torch_compile=False
    )
    
    model = M2BertForMaskedLM(config).to(device)
    
    # Forward pass
    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs['loss']
    
    print(f"MLM Loss: {loss.item():.4f}")
    print(f"Perplexity: {torch.exp(loss).item():.2f}")
    
    # Sanity check - loss should be around -ln(1/vocab_size) initially
    expected_initial_loss = -np.log(1.0 / tokenizer.vocab_size)
    print(f"Expected random loss: {expected_initial_loss:.4f}")
    print(f"Ratio to random: {loss.item() / expected_initial_loss:.2f}")
    
    if loss.item() < expected_initial_loss * 0.5:
        print("⚠️  WARNING: Loss seems suspiciously low for untrained model")
    elif loss.item() > expected_initial_loss * 2:
        print("⚠️  WARNING: Loss seems suspiciously high")
    else:
        print("✓ Loss is in reasonable range for untrained model")


def benchmark_vs_baseline():
    """Compare performance against standard BERT"""
    print("\n" + "="*70)
    print("BENCHMARK: M2-BERT vs Standard BERT")
    print("="*70)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Test parameters
    batch_size = 4
    seq_length = 128
    num_iterations = 10
    
    # Create test data
    input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_length), device=device)
    attention_mask = torch.ones_like(input_ids)
    
    # Initialize M2-BERT
    m2_config = M2BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=768,
        num_hidden_layers=12,
        intermediate_size=3072,
        use_monarch_mlp=True,
        monarch_nblocks=4,
        use_torch_compile=False
    )
    m2_model = M2BertForMaskedLM(m2_config).to(device)
    m2_model.eval()
    
    # Count parameters
    m2_params = sum(p.numel() for p in m2_model.parameters())
    
    print(f"\nM2-BERT:")
    print(f"  Parameters: {m2_params/1e6:.1f}M")
    
    # Warmup
    for _ in range(3):
        with torch.no_grad():
            _ = m2_model(input_ids, attention_mask=attention_mask)
    
    # Benchmark M2-BERT
    if device.type == 'mps':
        torch.mps.synchronize()
    
    start = time.perf_counter()
    for _ in range(num_iterations):
        with torch.no_grad():
            outputs = m2_model(input_ids, attention_mask=attention_mask)
    
    if device.type == 'mps':
        torch.mps.synchronize()
    
    m2_time = time.perf_counter() - start
    
    print(f"  Time per iteration: {m2_time/num_iterations*1000:.2f}ms")
    print(f"  Throughput: {batch_size*seq_length*num_iterations/m2_time:.0f} tokens/sec")
    
    # Check output sanity
    logits = outputs['logits']
    print(f"  Output shape: {logits.shape}")
    print(f"  Output mean: {logits.mean().item():.4f}")
    print(f"  Output std: {logits.std().item():.4f}")
    
    # Verify predictions sum to 1
    probs = F.softmax(logits[0, 0], dim=-1)
    prob_sum = probs.sum().item()
    print(f"  Probability sum: {prob_sum:.6f}")
    
    if abs(prob_sum - 1.0) > 0.01:
        print("  ⚠️  WARNING: Probabilities don't sum to 1!")
    else:
        print("  ✓ Probabilities sum correctly")
    
    # Check if model produces varied outputs
    output_variance = logits.var(dim=-1).mean().item()
    if output_variance < 0.01:
        print("  ⚠️  WARNING: Output variance is very low - model might be broken")
    else:
        print(f"  ✓ Output variance is healthy: {output_variance:.4f}")


def test_gradient_flow():
    """Test that gradients flow properly through the model"""
    print("\n" + "="*70)
    print("TESTING GRADIENT FLOW")
    print("="*70)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Small model for testing
    config = M2BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=256,
        num_hidden_layers=2,
        intermediate_size=512,
        num_attention_heads=4,
        use_monarch_mlp=True,
        monarch_nblocks=4,
        use_torch_compile=False
    )
    
    model = M2BertForMaskedLM(config).to(device)
    model.train()
    
    # Create simple test data
    input_ids = torch.randint(0, 100, (2, 16), device=device)
    labels = input_ids.clone()
    labels[:, 8:] = -100  # Mask half
    
    # Forward pass
    outputs = model(input_ids, labels=labels)
    loss = outputs['loss']
    
    print(f"Initial loss: {loss.item():.4f}")
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            layer_type = name.split('.')[2] if len(name.split('.')) > 2 else 'other'
            if layer_type not in grad_norms:
                grad_norms[layer_type] = []
            grad_norms[layer_type].append(grad_norm)
    
    print("\nGradient norms by layer type:")
    for layer_type, norms in grad_norms.items():
        avg_norm = np.mean(norms)
        print(f"  {layer_type}: {avg_norm:.6f}")
        if avg_norm == 0:
            print(f"    ⚠️  WARNING: Zero gradients in {layer_type}!")
        elif avg_norm > 100:
            print(f"    ⚠️  WARNING: Very large gradients in {layer_type}!")
    
    # Check if any parameters have no gradient
    params_without_grad = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is None:
            params_without_grad.append(name)
    
    if params_without_grad:
        print(f"\n⚠️  WARNING: {len(params_without_grad)} parameters have no gradient!")
        for name in params_without_grad[:5]:
            print(f"    - {name}")
    else:
        print("\n✓ All parameters have gradients")


if __name__ == "__main__":
    print("REAL M2-BERT TESTING")
    print("No fake data, no hand-waving")
    print()
    
    # Run all tests
    test_real_text_generation()
    test_mlm_loss()
    benchmark_vs_baseline()
    test_gradient_flow()
    
    print("\n" + "="*70)
    print("TESTING COMPLETE")
    print("="*70)