"""
M2-BERT Models
Apache 2.0 licensed foundation models with 32k context

Includes:
- Compatibility layer for pretrained weights
- Modern PyTorch implementations
- Monarch matrix optimizations
- Ray actor wrappers
"""

from .compatibility import M2BertModel, M2BertConfig, load_pretrained_m2bert
from .modern import M2BertModern
# from .monarch import MonarchMixerLayer, MonarchLinear  # TODO: Implement

__all__ = [
    'M2BertModel',
    'M2BertConfig', 
    'load_pretrained_m2bert',
    'M2BertModern',
    # 'MonarchMixerLayer',
    # 'MonarchLinear'
]