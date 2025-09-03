"""
Thea Code System
Actor-Centric Code Correction Architecture

Built from the journey:
- M2-BERT (Apache 2.0) for 32k context
- Facebook MBRL (MIT) for planning  
- Ray actors for distribution
- All of Poli's innovations combined
"""

__version__ = "1.0.0"
__author__ = "Thea Code Team"
__license__ = "Apache 2.0"

# Temporarily disabled while refactoring
# from .actors import *
# from .models import *
# from .utils import *

# Import only the new core components
try:
    from .core import *
except ImportError:
    pass  # Core not yet fully integrated