# Utils Module  
## Supporting Infrastructure and Libraries

This module contains all the supporting utilities, libraries, and components that enable our actor-centric architecture.

## 📁 Structure

### MBRL (`mbrl/`)
**License**: MIT ✅  
**Source**: Facebook Research (before archival)

Contains Facebook's production-tested Model-Based Reinforcement Learning library:

- **`models/`**: Ensemble dynamics models, gaussian MLPs, model environments
- **`planning/`**: CEM optimizers, trajectory optimization, SAC wrappers  
- **`util/`**: Replay buffers, math utilities, environment wrappers

**Why We Grabbed This**: 
Facebook archived this repo in Sept 2024, right when LFM2 was being developed. We got the MIT-licensed code before it disappeared into their commercial systems.

### Patterns (`patterns/`)
**License**: Our code, Apache 2.0  
**Source**: Battle-tested production patterns

Our reliable pattern-matching system for code fixes:

- **`eslint_patterns.py`**: Production-ready fix patterns  
- **`pattern_matcher.py`**: High-level pattern application interface

## 🧪 Facebook MBRL Integration

### What We Get from MBRL
1. **Ensemble Dynamics Models**: Proven uncertainty quantification
2. **CEM Planning**: Cross-Entropy Method optimization  
3. **Model Environments**: Simulated environments for planning
4. **Replay Buffers**: Efficient experience storage

### Key Components

#### EnsembleDynamicsModel
```python
from thea_code_system.utils.mbrl.models import EnsembleDynamicsModel

# Uncertainty-aware world modeling
dynamics_model = EnsembleDynamicsModel(
    ensemble_size=5,  # Multiple models for uncertainty
    obs_shape=(768,),  # M2-BERT hidden states
    action_shape=(100,),  # ESLint fix actions  
    device=device
)
```

#### CEM Optimizer
```python
from thea_code_system.utils.mbrl.planning import CEMOptimizer

# Plan sequences of code fixes
planner = CEMOptimizer(
    num_iterations=5,
    population_size=500,
    elite_ratio=0.1,
    device=device
)

# Find best action sequence
action_sequence = planner.optimize(state, model_env, horizon=10)
```

### The Cheetah Connection
The HalfCheetah environment we found in MBRL connects to Michael Poli's Neural Circuit Policies work! This is where:
- **NCPs** (19 neurons, C.elegans-inspired) were tested
- **Liquid time-constant networks** were validated  
- **Model-based RL** met **continuous control**

## 🎯 Pattern Matching System

### Design Philosophy
**ML Detects, Patterns Fix**: Neural networks identify problems, reliable patterns apply solutions.

### ESLint Patterns
Our battle-tested patterns with confidence scores:

```python
# High confidence (1.0) - mechanical fixes
"no-spaced-equals": {
    "regex": r"\s=\s=\s|\s!\s=\s", 
    "fix": lambda m: m.group().replace(" = = ", " == "),
    "reliability": 1.0
}

# Medium confidence (0.9) - contextual fixes  
"prefer-const": {
    "regex": r"\blet\s+(\w+)\s*=\s*([^;]+);(?![^{}]*\1\s*=)",
    "fix": lambda m: f"const {m.group(1)} = {m.group(2)};",
    "reliability": 0.9
}
```

### Pattern Application
```python
from thea_code_system.utils.patterns import PatternMatcher

matcher = PatternMatcher()

# Analyze code for potential fixes
analysis = matcher.analyze_code(code)

# Apply selected fixes
fixed_code, report = matcher.apply_fixes(code, rules=["no-spaced-equals"])
```

## 🔗 Integration Points

### With Actor System
```python
# Actors import utilities cleanly
from ...utils.mbrl.models import ModelEnv
from ...utils.patterns.eslint_patterns import CodePattern

# No scattered imports, clean architecture
```

### With Models
```python
# MBRL planning with M2-BERT states
obs = m2bert_actor.encode_code(code)  # Get M2-BERT representation
action = mbrl_planner.optimize(obs)   # Plan using MBRL
```

## 🏗️ The Complete Integration

### Our Frankenstein Architecture
We combine:

1. **M2-BERT** (Apache 2.0) for understanding
2. **Facebook MBRL** (MIT) for planning  
3. **Ray Actors** for distribution
4. **Pattern Matching** for reliability
5. **PyTorch Latest** for performance

### Why This Works
- **M2-BERT provides context**: 32k tokens of understanding
- **MBRL provides planning**: Multi-step fix sequences  
- **Patterns provide reliability**: Mechanical fixes that always work
- **Actors provide scale**: Horizontal distribution

## 📊 Production Benefits

### Reliability
- **Pattern confidence scores**: Know which fixes are safe
- **Ensemble uncertainty**: Multiple models provide confidence bounds
- **Graceful degradation**: Fall back to simpler fixes when uncertain

### Performance  
- **Batch processing**: Handle multiple files simultaneously
- **Memory efficiency**: Out-of-band tensor communication
- **Device optimization**: CPU/GPU/MPS automatic selection

### Scalability
- **Horizontal scaling**: Add more workers = more throughput  
- **Fault tolerance**: Individual component failures don't crash system
- **Resource sharing**: Efficient GPU and memory usage

## 🎭 The Poli Universe Connection

### What We Discovered
Through our journey, we found Michael Poli's fingerprints everywhere:

1. **Liquid Networks** → adaptive timing  
2. **Neural Circuit Policies** → C.elegans control (tested on Cheetah!)
3. **M2-BERT** → Monarch matrices  
4. **MAD** → modular attention
5. **Liquid Foundation Models** → the culmination

### Our Realization
We accidentally recreated their methodology:
- **JSON outputs** ✓ (our fix format)
- **Up/down voting** ✓ (pattern confidence)  
- **Teacher models** ✓ (fallback to larger models)
- **32k context** ✓ (inherited from M2-BERT)

## 🚀 Future Extensions

### Monitoring (`monitoring/`)
- Real-time performance metrics
- Actor health monitoring  
- Resource usage tracking
- Alert systems

### Caching (`caching/`)  
- Intelligent result caching
- Distributed cache coordination
- Cache invalidation strategies

### Streaming (`streaming/`)
- Real-time code processing
- WebSocket integration
- Event-driven updates

## 🔐 Legal Status

### Open Source Foundation
- ✅ **Facebook MBRL**: MIT License (permissive)
- ✅ **Our Patterns**: Apache 2.0 (our code)
- ✅ **Integration Code**: Apache 2.0 (our code)

### Commercial Use
- ✅ Can be used in commercial products
- ✅ Can be modified and redistributed
- ✅ No copyleft restrictions
- ✅ Future-proof licensing

The utils module provides the production-ready foundation that makes our actor-centric architecture possible.