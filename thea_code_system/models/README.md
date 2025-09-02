# Models Module
## AI Models and Neural Architectures

This module contains all the AI models and neural network architectures used in our system, carefully organized and legally compliant.

## 📁 Structure

### M2-BERT (`m2bert/`)
**License**: Apache 2.0 ✅  
**Source**: togethercomputer/m2-bert-80M-32k-retrieval

Contains our implementation and compatibility layers for M2-BERT:

- **`compatibility.py`**: Loads pretrained weights, handles model format differences
- **`modern.py`**: Modern PyTorch implementation with latest optimizations  
- **`monarch.py`**: Monarch matrix implementations for O(n^3/2) complexity

**Key Changes from Original Files**:
- ✅ Cleaned up imports and file organization
- ✅ Added proper error handling for missing model files
- ✅ Enhanced configuration loading with fallbacks
- ✅ Production-ready device management

### Liquid (`liquid/`)
**License**: Research patterns, our implementation  
**Source**: Liquid AI research papers and blog posts

Implementations of patterns from Liquid Foundation Models:

- **Hybrid conv-attention architectures**
- **Double-gated convolutions** 
- **Hardware-aware optimizations**
- **32k context processing**

## 🧠 The M2-BERT Foundation

### Why M2-BERT?
1. **Apache 2.0 License**: Completely free to use and modify
2. **32k Context**: Process entire files in single pass
3. **Proven Architecture**: Monarch matrices provide O(n^3/2) complexity
4. **Pretrained Weights**: Available on HuggingFace

### Loading Pretrained Models
```python
from thea_code_system.models.m2bert import load_pretrained_m2bert

model, config = load_pretrained_m2bert(
    "togethercomputer/m2-bert-80M-32k-retrieval"
)
```

### Compatibility Layer
The compatibility layer handles:
- **Weight format differences** between original and our implementation
- **Missing components** (Hyena operators, FlashFFT)
- **Configuration mapping** from HuggingFace format
- **Graceful fallbacks** when components are missing

## 🌊 Liquid Patterns

### What We Learned from LFM2
The Liquid Foundation Model architecture taught us:

1. **Hybrid is better than pure**: Conv + Attention > Pure Attention
2. **Hardware awareness matters**: Different block orderings for different devices
3. **32k context proves lineage**: M2-BERT → MAD → LFM2 all use 32k
4. **Multiplicative gating**: B*x and C*x instead of additive combinations

### Our Implementation
```python
# LFM2-style double-gated convolution
def lfm2_conv_block(x):
    projected = input_proj(x)
    x, B, C = projected.chunk(3, dim=-1)
    
    x = B * x           # First gate
    x = conv(x)         # Short convolution  
    x = C * x           # Second gate
    
    return output_proj(x)
```

## 🔬 The Complete Research Lineage

### Michael Poli's Ecosystem
Through our journey, we discovered Poli's work spans:

1. **Liquid Time-Constant Networks**: Adaptive timing neurons
2. **Neural Circuit Policies**: C.elegans-inspired control (19 neurons!)
3. **M2-BERT Monarch Matrices**: Subquadratic attention
4. **MAD (Modular Attention)**: Bolt-on pretrained modules  
5. **Liquid Foundation Models**: The culmination

### Our Realization
We inadvertently recreated their training methodology:
- **JSON structured outputs** (our ESLint corrections)
- **Up/down voting** (our pattern reliability scores)
- **Teacher model guidance** (falling back to larger models)
- **32k context processing** (entire files)

## 🛠️ Production Optimizations

### PyTorch Latest Features
```python
# Automatic optimizations
if hasattr(torch, 'compile'):
    model = torch.compile(model, mode="max-autotune")

# Native attention when available  
attn_output = F.scaled_dot_product_attention(
    query, key, value,
    attn_mask=attention_mask
)
```

### Device Management
```python
# Automatic device selection with optimizations
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
    torch.set_num_threads(8)
```

## 📊 Model Statistics

### M2-BERT-32k
- **Parameters**: 80M (efficient!)
- **Context Length**: 32,768 tokens
- **Architecture**: Monarch Mixer with butterfly transforms
- **Complexity**: O(n^3/2) vs O(n^2) for standard attention

### Performance Characteristics
- **CPU Inference**: Excellent with torch.compile
- **Memory Usage**: Linear with sequence length (not quadratic)
- **Batch Processing**: Efficient parallel processing
- **32k Context**: Handles entire files without chunking

## 🎯 Integration with Actors

The models are designed to work seamlessly with our actor system:

```python
@ray.remote(num_gpus=0.25)
class ModelActor:
    def __init__(self):
        self.model, self.config = load_pretrained_m2bert(...)
        self.model = torch.compile(self.model)  # Optimize
    
    async def process_code(self, code: str):
        # 32k context processing
        return self.model(tokenize(code, max_length=32768))
```

## 🔐 Legal Compliance

### Licenses
- ✅ **M2-BERT**: Apache 2.0 (commercial use allowed)
- ✅ **Our Code**: Apache 2.0 (open source)
- ✅ **Research Patterns**: Not copyrightable (mathematical formulations)

### What We Can Do
- ✅ Use commercially
- ✅ Modify and redistribute  
- ✅ Create derivative works
- ✅ Use in proprietary systems

## 🚀 Future Extensions

- **Model Quantization**: 4-bit, 8-bit variants for edge deployment
- **GGUF Export**: For llama.cpp ecosystem compatibility
- **LoRA Fine-tuning**: Efficient adaptation to specific domains
- **Knowledge Distillation**: Train smaller models from M2-BERT

The models module provides the AI foundation for reliable, scalable code correction.