# Actor System
## Ray-Based Distributed Processing

The actor system is the heart of our distributed architecture. Every component is designed as a Ray actor for maximum scalability and fault tolerance.

## 🎭 Core Actors

### CodeCorrectionActor
**Location**: `core/correction_actor.py`

The main workhorse actor that:
- Loads M2-BERT-32k for context understanding
- Applies reliable pattern-based fixes
- Handles 32k token context windows
- Provides async processing throughout

**Key Changes from Original Design**:
- ✅ Proper import paths for new structure
- ✅ Production error handling 
- ✅ Health check capabilities
- ✅ Device optimization (CPU/GPU/MPS)

### OrchestrationActor  
**Location**: `core/orchestration_actor.py`

The conductor that:
- Manages worker actor pools
- Implements round-robin load balancing
- Handles fault tolerance and health checks
- Processes entire codebases efficiently

**Key Changes**:
- ✅ Comprehensive health monitoring
- ✅ Batch processing optimization
- ✅ Detailed reporting system
- ✅ Error recovery mechanisms

### TensorStoreActor
**Location**: `core/tensor_store_actor.py`

Enables efficient tensor sharing:
- Out-of-band tensor communication
- Automatic memory management
- Zero-copy operations when possible
- Storage statistics and monitoring

**Key Innovation**: 
This solves the Ray serialization bottleneck for large tensors. Instead of passing tensors through Ray's object store, we use dedicated tensor storage.

## 🏗️ Architecture Principles

### 1. Async Everywhere
```python
@ray.remote
class CodeCorrectionActor:
    async def analyze_code(self, code: str) -> Dict:
        # All methods are async for maximum concurrency
```

### 2. Fault Tolerance
```python
async def health_check(self) -> Dict[str, Any]:
    # Every actor provides health checks
    # Orchestrator monitors and handles failures
```

### 3. Resource Management
```python
@ray.remote(num_gpus=0.25 if torch.cuda.is_available() else 0)
class CodeCorrectionActor:
    # Proper GPU allocation and sharing
```

### 4. Out-of-Band Communication
```python
# Instead of:
result = await actor.process.remote(large_tensor)  # Slow serialization

# We do:
tensor_id = await tensor_store.put.remote(large_tensor)  # Fast reference
result = await actor.process.remote(tensor_id)  # Just pass ID
```

## 🔧 Integration Points

### With M2-BERT Models
```python
from ...models.m2bert.compatibility import load_pretrained_m2bert

# Clean import paths, no more scattered files
```

### With MBRL Components  
```python
from ...utils.mbrl.models import ModelEnv, EnsembleDynamicsModel

# Facebook's production-tested MBRL components
```

### With Pattern Matching
```python
from ...utils.patterns.eslint_patterns import CodePattern

# Battle-tested fix patterns
```

## 📈 Performance Characteristics

- **Horizontal Scaling**: Add more actors = more throughput
- **Fault Tolerance**: Individual actor failures don't crash system
- **Memory Efficiency**: Out-of-band tensor storage prevents OOM
- **GPU Sharing**: Multiple actors can share GPU resources

## 🧪 What We Learned

### From the Journey
1. **Started**: Cleaning emoji-filled BERT code
2. **Discovered**: M2-BERT's Apache 2.0 license
3. **Built**: Ray actor architecture for distribution
4. **Realized**: We recreated Liquid AI's methodology

### Key Insights
- **Actor-first design** enables true distributed ML
- **Out-of-band communication** solves serialization bottlenecks  
- **Health monitoring** is critical for production systems
- **Pattern-based fixes** provide reliability that pure ML cannot

## 🚀 Future Extensions

- **Training Actors**: For online learning and model updates
- **Monitoring Actors**: For system-wide observability  
- **Cache Actors**: For intelligent result caching
- **Stream Actors**: For real-time code processing

The actor system provides the foundation for any distributed ML workload, not just code correction.