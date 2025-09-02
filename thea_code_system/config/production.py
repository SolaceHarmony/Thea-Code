#!/usr/bin/env python
"""
Production Configuration
Centralized configuration for the entire system

This contains all the settings needed for production deployment
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import os


@dataclass
class ProductionConfig:
    """
    Production configuration for Thea Code System
    
    All settings needed for production deployment
    """
    
    # Model settings
    model_path: str = "togethercomputer/m2-bert-80M-32k-retrieval"
    context_length: int = 32768  # 32k context!
    hidden_size: int = 768
    
    # Actor system settings
    max_workers: int = 8
    enable_torch_compile: bool = True
    confidence_threshold: float = 0.7
    
    # Performance settings
    batch_size: int = 4
    max_fixes_per_file: int = 10
    
    # Supported ESLint rules
    supported_rules: List[str] = field(default_factory=lambda: [
        "no-spaced-equals",
        "arrow-spacing", 
        "prefer-const",
        "prefer-async-await",
        "import-order",
        "no-unused-vars",
        "semicolon",
        "quotes",
        "indent",
        "no-trailing-spaces",
        "no-multiple-spaces"
    ])
    
    # MBRL settings
    mbrl_ensemble_size: int = 5
    mbrl_planning_horizon: int = 10
    mbrl_population_size: int = 500
    
    # Hardware settings  
    device_preference: List[str] = field(default_factory=lambda: ["cuda", "mps", "cpu"])
    gpu_memory_fraction: float = 0.8
    cpu_threads: Optional[int] = None  # Auto-detect
    
    # Monitoring settings
    enable_health_checks: bool = True
    health_check_interval: int = 60  # seconds
    max_tensor_store_gb: float = 4.0
    
    # Logging settings
    log_level: str = "INFO"
    log_actor_performance: bool = True
    log_pattern_usage: bool = True
    
    def __post_init__(self):
        """Validate and adjust settings"""
        
        # Auto-detect CPU threads if not specified
        if self.cpu_threads is None:
            self.cpu_threads = min(8, os.cpu_count() or 4)
        
        # Validate context length
        if self.context_length > 32768:
            print("⚠️  Warning: Context length > 32k may not work with M2-BERT")
        
        # Validate confidence threshold
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'model_path': self.model_path,
            'context_length': self.context_length,
            'max_workers': self.max_workers,
            'enable_torch_compile': self.enable_torch_compile,
            'supported_rules': self.supported_rules,
            'confidence_threshold': self.confidence_threshold
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ProductionConfig':
        """Create from dictionary"""
        return cls(**config_dict)
    
    @classmethod  
    def for_development(cls) -> 'ProductionConfig':
        """Configuration optimized for development"""
        return cls(
            max_workers=2,
            enable_torch_compile=False,  # Faster startup
            context_length=8192,  # Smaller for testing
            confidence_threshold=0.5,  # More lenient
            log_level="DEBUG"
        )
    
    @classmethod
    def for_production(cls) -> 'ProductionConfig':
        """Configuration optimized for production"""  
        return cls(
            max_workers=16,
            enable_torch_compile=True,  # Maximum performance
            context_length=32768,  # Full context
            confidence_threshold=0.8,  # Higher confidence
            log_level="INFO"
        )
    
    @classmethod
    def for_edge_device(cls) -> 'ProductionConfig':
        """Configuration for edge/mobile devices"""
        return cls(
            max_workers=2,
            enable_torch_compile=False,  # Compatibility
            context_length=4096,  # Reduced for memory
            confidence_threshold=0.9,  # Only high-confidence fixes
            device_preference=["cpu"],  # CPU only
            log_level="WARNING"
        )


# Convenience functions for common configurations
def get_config(environment: str = "production") -> ProductionConfig:
    """Get configuration for specific environment"""
    
    configs = {
        "development": ProductionConfig.for_development,
        "dev": ProductionConfig.for_development,
        "production": ProductionConfig.for_production,
        "prod": ProductionConfig.for_production,
        "edge": ProductionConfig.for_edge_device,
        "mobile": ProductionConfig.for_edge_device
    }
    
    if environment not in configs:
        print(f"⚠️  Unknown environment '{environment}', using production config")
        environment = "production"
    
    return configs[environment]()


def load_config_from_env() -> ProductionConfig:
    """Load configuration from environment variables"""
    
    config = ProductionConfig()
    
    # Override with environment variables if present
    if os.getenv("THEA_MODEL_PATH"):
        config.model_path = os.getenv("THEA_MODEL_PATH")
    
    if os.getenv("THEA_MAX_WORKERS"):
        config.max_workers = int(os.getenv("THEA_MAX_WORKERS"))
    
    if os.getenv("THEA_CONTEXT_LENGTH"):
        config.context_length = int(os.getenv("THEA_CONTEXT_LENGTH"))
    
    if os.getenv("THEA_CONFIDENCE_THRESHOLD"):
        config.confidence_threshold = float(os.getenv("THEA_CONFIDENCE_THRESHOLD"))
    
    if os.getenv("THEA_LOG_LEVEL"):
        config.log_level = os.getenv("THEA_LOG_LEVEL")
    
    return config


# Export for easy importing
__all__ = [
    'ProductionConfig',
    'get_config', 
    'load_config_from_env'
]