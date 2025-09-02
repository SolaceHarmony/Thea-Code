#!/usr/bin/env python3
"""
Test the REAL M2-BERT-32k model from HuggingFace
This is what we should have done from the start
"""

import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig
import json
import os

def test_m2bert_32k():
    """Test the actual pretrained M2-BERT with 32k context"""
    print("="*70)
    print("M2-BERT-32K FROM HUGGINGFACE")
    print("="*70)
    
    model_path = "./m2_models/models--togethercomputer--m2-bert-80M-32k-retrieval/snapshots/a2ccdc5b5661a282c77545e586a019f387ab7a48"
    
    # Load config
    config_path = os.path.join(model_path, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    
    print("\nModel Configuration:")
    print(f"  Architecture: {config.get('architectures', ['Unknown'])[0]}")
    print(f"  Hidden size: {config.get('hidden_size', 'N/A')}")
    print(f"  Layers: {config.get('num_hidden_layers', 'N/A')}")
    print(f"  Max position embeddings: {config.get('max_position_embeddings', 'N/A')}")
    print(f"  Attention heads: {config.get('num_attention_heads', 'N/A')}")
    
    # Check for Monarch-specific config
    if 'use_monarch_mlp' in config:
        print(f"  Use Monarch MLP: {config['use_monarch_mlp']}")
    if 'monarch_mlp_nblocks' in config:
        print(f"  Monarch blocks: {config['monarch_mlp_nblocks']}")
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Try to load the model
    print("Loading model...")
    try:
        # First try AutoModel
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        print("✓ Model loaded successfully!")
    except Exception as e:
        print(f"Error loading with AutoModel: {e}")
        
        # Try loading the state dict directly
        print("\nTrying to load state dict directly...")
        model_file = os.path.join(model_path, "pytorch_model.bin")
        if os.path.exists(model_file):
            state_dict = torch.load(model_file, map_location='cpu')
            print(f"State dict keys: {list(state_dict.keys())[:5]}...")
            print(f"Total keys: {len(state_dict)}")
            
            # Check structure
            for key in list(state_dict.keys())[:10]:
                print(f"  {key}: {state_dict[key].shape}")
        return
    
    # Move to device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params/1e6:.1f}M")
    
    # Test with real text
    test_texts = [
        "The capital of France is Paris.",
        "Machine learning models can process large amounts of data efficiently.",
        "Python is widely used in data science and artificial intelligence.",
    ]
    
    print("\nTesting embeddings:")
    for text in test_texts:
        print(f"\nInput: {text[:50]}...")
        
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            
            # Handle different output formats
            if hasattr(outputs, 'last_hidden_state'):
                embeddings = outputs.last_hidden_state
            elif hasattr(outputs, 'hidden_states'):
                embeddings = outputs.hidden_states[-1]
            else:
                embeddings = outputs
        
        # Pool embeddings (mean pooling)
        attention_mask = inputs['attention_mask'].unsqueeze(-1)
        masked_embeddings = embeddings * attention_mask
        summed = torch.sum(masked_embeddings, dim=1)
        counted = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
        mean_pooled = summed / counted
        
        print(f"  Embedding shape: {mean_pooled.shape}")
        print(f"  Embedding norm: {mean_pooled.norm(dim=-1).item():.4f}")
        print(f"  First 5 values: {mean_pooled[0, :5].cpu().numpy()}")


def test_long_context():
    """Test with actual long context"""
    print("\n" + "="*70)
    print("TESTING LONG CONTEXT (32K)")
    print("="*70)
    
    model_path = "./m2_models/models--togethercomputer--m2-bert-80M-32k-retrieval/snapshots/a2ccdc5b5661a282c77545e586a019f387ab7a48"
    
    # Create a long text
    long_text = """
    Monarch matrices represent a groundbreaking advancement in efficient transformer architectures.
    By decomposing weight matrices using butterfly transforms and block-diagonal structures,
    they achieve subquadratic complexity while maintaining model quality.
    """ * 100  # Repeat to make it long
    
    print(f"Text length: {len(long_text)} characters")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Tokenize with different lengths
        for max_length in [512, 2048, 8192]:
            inputs = tokenizer(
                long_text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding='max_length'
            )
            
            print(f"\nMax length {max_length}:")
            print(f"  Input shape: {inputs['input_ids'].shape}")
            print(f"  Actual tokens: {inputs['attention_mask'].sum().item()}")
            
    except Exception as e:
        print(f"Error: {e}")


def analyze_model_structure():
    """Analyze the actual model structure"""
    print("\n" + "="*70)
    print("MODEL STRUCTURE ANALYSIS")
    print("="*70)
    
    model_file = "./m2_models/models--togethercomputer--m2-bert-80M-32k-retrieval/snapshots/a2ccdc5b5661a282c77545e586a019f387ab7a48/pytorch_model.bin"
    
    if os.path.exists(model_file):
        state_dict = torch.load(model_file, map_location='cpu')
        
        # Analyze layer structure
        layer_types = {}
        for key in state_dict.keys():
            parts = key.split('.')
            if len(parts) > 2:
                layer_type = parts[2]
                if layer_type not in layer_types:
                    layer_types[layer_type] = []
                layer_types[layer_type].append(key)
        
        print("Layer types found:")
        for layer_type, keys in layer_types.items():
            print(f"  {layer_type}: {len(keys)} parameters")
        
        # Check for Monarch-specific parameters
        monarch_params = [k for k in state_dict.keys() if 'monarch' in k.lower() or 'block' in k.lower()]
        if monarch_params:
            print(f"\nMonarch-specific parameters: {len(monarch_params)}")
            for param in monarch_params[:5]:
                print(f"  {param}")
        
        # Check weight shapes to detect block structure
        print("\nChecking for block-diagonal structure:")
        for key, tensor in list(state_dict.items())[:20]:
            if 'weight' in key and tensor.dim() == 3:
                print(f"  {key}: shape={tensor.shape} (likely block-diagonal!)")
            elif 'weight' in key and tensor.dim() == 2 and tensor.shape[0] != tensor.shape[1]:
                # Check if it could be reshaped to blocks
                h, w = tensor.shape
                for nblocks in [2, 4, 8, 16]:
                    if h % nblocks == 0 and w % nblocks == 0:
                        print(f"  {key}: shape={tensor.shape} (could be {nblocks} blocks)")
                        break


if __name__ == "__main__":
    print("TESTING REAL M2-BERT-32K FROM HUGGINGFACE")
    print("This is the actual pretrained model")
    print()
    
    test_m2bert_32k()
    test_long_context()
    analyze_model_structure()
    
    print("\n" + "="*70)
    print("DONE - Now we know what we're working with!")
    print("="*70)