"""
Moveworks Reasoning Engine Configuration.
Provides configuration-driven approach for all reasoning parameters.
"""

import os
from typing import Optional
from dataclasses import dataclass


@dataclass
class ReasoningConfig:
    """Configuration for Moveworks Reasoning Engine components."""
    
    # Planning Iteration Loop Configuration
    planning_iterations_max: int = 3
    
    # Execution Iteration Loop Configuration  
    execution_steps_max: int = 5
    
    # Multi-Plugin Response Configuration
    max_parallel_plugins: int = 3
    response_timeout: float = 30.0
    
    # Memory Management Configuration
    memory_window_size: int = 20  # Maximum messages to keep
    memory_window_min: int = 6    # Minimum context for reasoning
    
    @classmethod
    def from_environment(cls) -> 'ReasoningConfig':
        """Load configuration from environment variables."""
        return cls(
            planning_iterations_max=int(os.getenv('PLANNING_ITERATIONS_MAX', '3')),
            execution_steps_max=int(os.getenv('EXECUTION_STEPS_MAX', '5')),
            max_parallel_plugins=int(os.getenv('MAX_PARALLEL_PLUGINS', '3')),
            response_timeout=float(os.getenv('RESPONSE_TIMEOUT', '30.0')),
            memory_window_size=int(os.getenv('MEMORY_WINDOW_SIZE', '20')),
            memory_window_min=int(os.getenv('MEMORY_WINDOW_MIN', '6'))
        )
    
    def validate(self) -> list[str]:
        """Validate configuration values."""
        errors = []
        
        if self.planning_iterations_max < 1:
            errors.append("planning_iterations_max must be >= 1")
        
        if self.execution_steps_max < 1:
            errors.append("execution_steps_max must be >= 1")
            
        if self.max_parallel_plugins < 1:
            errors.append("max_parallel_plugins must be >= 1")
            
        if self.response_timeout <= 0:
            errors.append("response_timeout must be > 0")
            
        if self.memory_window_size < self.memory_window_min:
            errors.append("memory_window_size must be >= memory_window_min")
            
        if self.memory_window_min < 1:
            errors.append("memory_window_min must be >= 1")
            
        return errors


# Global configuration instance
_reasoning_config: Optional[ReasoningConfig] = None


def get_reasoning_config() -> ReasoningConfig:
    """Get the global reasoning configuration instance."""
    global _reasoning_config
    
    if _reasoning_config is None:
        _reasoning_config = ReasoningConfig.from_environment()
        
        # Validate configuration
        errors = _reasoning_config.validate()
        if errors:
            raise ValueError(f"Invalid reasoning configuration: {', '.join(errors)}")
    
    return _reasoning_config


def reload_reasoning_config() -> ReasoningConfig:
    """Reload configuration from environment (useful for testing)."""
    global _reasoning_config
    _reasoning_config = None
    return get_reasoning_config()
