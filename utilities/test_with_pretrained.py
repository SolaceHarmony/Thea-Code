#!/usr/bin/env python3
"""
Test M2-BERT with actual pretrained weights from HuggingFace
This is the real test - no excuses
"""

import torch
import torch.nn as nn
from transformers import BertTokenizer
from huggingface_hub import hf_hub_download
import sys
import os

# Add HazyResearch code to path
sys.path.append('hazyresearch_m2/bert')

def download_pretrained_weights():
    """Download pretrained M2-BERT weights from HuggingFace"""
    print("Downloading pretrained M2-BERT-80M weights from HuggingFace...")
    
    try:
        # Download the model file
        model_path = hf_hub_download(
            repo_id="danfu09/m2-bert-80m",
            filename="pytorch_model.bin",
            cache_dir="./pretrained_models"
        )
        print(f"Downloaded to: {model_path}")
        return model_path
    except Exception as e:
        print(f"Error downloading from HuggingFace: {e}")
        
        # Try direct download from Google Storage
        print("\nTrying direct download from Google Storage...")
        import urllib.request
        
        os.makedirs("pretrained_models", exist_ok=True)
        url = "https://storage.googleapis.com/danfu-data/checkpoints/bert/pretrained_M2-BERT_checkpoints/m2-bert-128.pt"
        local_path = "pretrained_models/m2-bert-128.pt"
        
        try:
            urllib.request.urlretrieve(url, local_path)
            print(f"Downloaded to: {local_path}")
            return local_path
        except Exception as e2:
            print(f"Error with direct download: {e2}")
            return None


def load_hazyresearch_model():
    """Load the actual HazyResearch M2-BERT model"""
    print("\nLoading HazyResearch M2-BERT implementation...")
    
    try:
        from src.create_bert import create_bert_mlm
        from omegaconf import OmegaConf
        
        # Load config from their YAML
        yaml_path = "hazyresearch_m2/bert/yamls/finetune-glue/monarch-mixer-finetune-glue-768dim-80m-parameters.yaml"
        
        if os.path.exists(yaml_path):
            with open(yaml_path) as f:
                config = OmegaConf.load(f)
            
            print("Config loaded successfully")
            
            # Create model
            model = create_bert_mlm(
                pretrained_model_name='bert-base-uncased',
                model_config=config.model.get('model_config', {}),
                tokenizer_name='bert-base-uncased'
            )
            
            return model
        else:
            print(f"Config file not found: {yaml_path}")
            return None
            
    except ImportError as e:
        print(f"Import error: {e}")
        print("Installing required dependencies...")
        os.system("pip install omegaconf einops")
        return None


def test_with_real_weights():
    """Test with actual pretrained weights"""
    print("="*70)
    print("TESTING WITH PRETRAINED WEIGHTS")
    print("="*70)
    
    # Download weights
    weights_path = download_pretrained_weights()
    if not weights_path:
        print("Failed to download weights")
        return
    
    # Load the actual model
    model = load_hazyresearch_model()
    if model is None:
        print("\nFalling back to our implementation...")
        # Use our implementation instead
        from m2bert_official import M2BertForMaskedLM, M2BertConfig
        
        config = M2BertConfig(
            hidden_size=768,
            num_hidden_layers=12,
            intermediate_size=3072,
            use_monarch_mlp=True,
            monarch_mlp_nblocks=4
        )
        model = M2BertForMaskedLM(config)
    
    # Try to load weights
    try:
        print(f"\nLoading weights from: {weights_path}")
        state_dict = torch.load(weights_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'model' in state_dict:
            state_dict = state_dict['model']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        
        # Try to load with mismatched keys allowed
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        print(f"Loaded weights!")
        if missing_keys:
            print(f"  Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            print(f"  Unexpected keys: {len(unexpected_keys)}")
            
    except Exception as e:
        print(f"Error loading weights: {e}")
        print("Testing with random weights instead...")
    
    # Move to device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Test with real text
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    test_sentences = [
        "The capital of France is [MASK].",
        "Machine learning is a type of [MASK] intelligence.",
        "The [MASK] jumped over the lazy dog.",
    ]
    
    print("\nTesting predictions:")
    for sentence in test_sentences:
        print(f"\nInput: {sentence}")
        
        inputs = tokenizer(sentence, return_tensors="pt")
        input_ids = inputs['input_ids'].to(device)
        
        # Find mask position
        mask_pos = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1][0]
        
        # Get predictions
        with torch.no_grad():
            if hasattr(model, 'bert'):
                # HazyResearch model structure
                outputs = model(input_ids)
                logits = outputs
            else:
                # Our model structure
                outputs = model(input_ids)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs
        
        # Get top predictions
        mask_logits = logits[0, mask_pos]
        top_5 = torch.topk(mask_logits, 5)
        
        print("  Top 5 predictions:")
        for i, token_id in enumerate(top_5.indices):
            token = tokenizer.decode([token_id])
            prob = torch.softmax(mask_logits, dim=-1)[token_id].item()
            print(f"    {i+1}. '{token}' (prob: {prob:.3f})")


def compare_implementations():
    """Compare our implementation with HazyResearch's"""
    print("\n" + "="*70)
    print("COMPARING IMPLEMENTATIONS")
    print("="*70)
    
    from m2bert_official import M2BertForMaskedLM, M2BertConfig
    
    # Our implementation
    our_config = M2BertConfig(
        hidden_size=768,
        num_hidden_layers=12,
        intermediate_size=3072,
        use_monarch_mlp=True,
        monarch_mlp_nblocks=4
    )
    our_model = M2BertForMaskedLM(our_config)
    
    # Count parameters
    our_params = sum(p.numel() for p in our_model.parameters())
    
    print(f"Our implementation:")
    print(f"  Total parameters: {our_params/1e6:.1f}M")
    
    # Check layer structure
    print(f"\nLayer structure:")
    for name, module in our_model.named_modules():
        if 'Linear' in module.__class__.__name__:
            if hasattr(module, 'weight'):
                print(f"  {name}: {module.__class__.__name__} {tuple(module.weight.shape)}")


if __name__ == "__main__":
    print("REAL TEST WITH PRETRAINED WEIGHTS")
    print("No excuses - this should work or we know what's broken")
    print()
    
    test_with_real_weights()
    compare_implementations()
    
    print("\n" + "="*70)
    print("REALITY CHECK COMPLETE")
    print("="*70)