import torch
import gc
import time
import logging
import psutil
import os
from typing import Dict, Any, Optional, Set

logger = logging.getLogger(__name__)


class MemoryManager:
    """Manages memory for training to avoid OOM errors"""
    
    def __init__(self, target_usage: float = 0.85, warning_threshold: float = 0.92):
        """
        Initialize memory manager
        
        Args:
            target_usage: Target GPU memory usage (0.0-1.0)
            warning_threshold: Threshold to trigger warnings
        """
        self.target_usage = target_usage
        self.warning_threshold = warning_threshold
        self.last_check = time.time()
        self.check_interval = 5  # seconds
        self.edge_memory_threshold = 4000  # MB - adjust based on your edge device
        self.memory_logger = logging.getLogger('memory_tracker')

    def enhance_memory_logging(self) -> Dict[str, Any]:
        """Add edge device specific metrics"""
        edge_memory_metrics = {
            "peak_memory_mb": torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0,
            "memory_fragmentation": self._calculate_fragmentation(),
            "batch_memory_profile": self._profile_batch_memory_usage(),
            "inference_memory_timeline": self._track_inference_memory_timeline(),
            "model_size_mb": self._get_model_size_mb(),
            "activation_memory_mb": self._estimate_activation_memory(),
            "quantization_memory_savings": self._quantization_memory_savings(),
        }
    
        # Log critical edge metrics
        for metric_name, value in edge_memory_metrics.items():
            self.memory_logger.info(f"Edge metric - {metric_name}: {value}")
            
        # Set alerts for memory thresholds specific to edge deployment
        if edge_memory_metrics["peak_memory_mb"] > self.edge_memory_threshold:
            self.memory_logger.warning(
                f"Edge memory threshold exceeded: {edge_memory_metrics['peak_memory_mb']:.2f}MB > {self.edge_memory_threshold}MB"
            )
        return edge_memory_metrics
    
    def _calculate_fragmentation(self) -> float:
        """Calculate memory fragmentation ratio"""
        if not torch.cuda.is_available():
            return 0.0
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        if reserved == 0:
            return 0.0
        return 1.0 - (allocated / reserved)
    
    def _profile_batch_memory_usage(self) -> Dict[str, float]:
        """Profile memory usage for a typical batch"""
        if not torch.cuda.is_available():
            return {"current_batch_mb": 0, "reserved_mb": 0, "utilization_ratio": 0}
        
        current_allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        return {
            "current_batch_mb": current_allocated,
            "reserved_mb": reserved,
            "utilization_ratio": current_allocated / max(1, reserved)
        }
    
    def _track_inference_memory_timeline(self) -> Dict[str, float]:
        """Track memory usage during inference steps"""
        if not torch.cuda.is_available():
            return {"current_mb": 0}
        current = torch.cuda.memory_allocated() / 1024**2
        return {"current_mb": current}
    
    def _get_model_size_mb(self) -> float:
        """Calculate model size in MB"""
        if not torch.cuda.is_available():
            return 0
        return torch.cuda.memory_allocated() / 1024**2
    
    def _estimate_activation_memory(self) -> float:
        """Estimate activation memory based on model architecture"""
        if not torch.cuda.is_available():
            return 0
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        return reserved - allocated
    
    def _quantization_memory_savings(self) -> float:
        """Calculate memory savings from quantization"""
        if not torch.cuda.is_available():
            return 0
        # Estimate based on bit reduction (32-bit to 4/8-bit is 4x-8x saving)
        current_size = torch.cuda.memory_allocated() / 1024**2
        estimated_savings = current_size * 0.7  # Assuming ~70% savings from quantization
        return estimated_savings
    
    def should_check(self) -> bool:
        """Check if we should monitor memory"""
        now = time.time()
        if now - self.last_check > self.check_interval:
            self.last_check = now
            return True
        return False
    
    def current_usage(self) -> float:
        """Get current GPU memory usage percentage"""
        if not torch.cuda.is_available():
            return 0.0
            
        device = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(device)
        total = torch.cuda.get_device_properties(device).total_memory
        return allocated / total
    
    def is_memory_critical(self) -> bool:
        """Check if memory usage is at critical levels"""
        if not self.should_check():
            return False
            
        usage = self.current_usage()
        if usage > self.warning_threshold:
            logger.warning(f"Critical GPU memory usage: {usage:.1%}")
            torch.cuda.empty_cache()
            gc.collect()
            return True
        return False
    
    def suggest_batch_reduction(self) -> int:
        """Suggest batch size reduction factor based on memory pressure"""
        usage = self.current_usage()
        if usage > self.target_usage:
            # Calculate reduction factor based on how far we are above target
            reduction = usage / self.target_usage
            return max(1, int(reduction * 2))
        return 1


def deep_cleanup():
    """Perform aggressive memory cleanup"""
    # Clear CUDA cache
    torch.cuda.empty_cache()
    
    # Clear CPU memory
    gc.collect()
    
    # Release memory from PyTorch's memory pool
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Print memory stats for monitoring
    if torch.cuda.is_available():
        logger.info(
            f"Memory cleanup: {torch.cuda.memory_allocated()/1024**2:.2f}MB allocated, "
            f"{torch.cuda.memory_reserved()/1024**2:.2f}MB reserved"
        )


def safe_set_training_mode(model: torch.nn.Module, mode: bool = True, visited: Set = None, depth: int = 0):
    """Safely sets training mode by avoiding infinite recursion."""
    if visited is None:
        visited = set()
    
    if depth > 100:  # Safety limit
        return
        
    module_id = id(model)
    if module_id in visited:
        return
    visited.add(module_id)
    
    try:
        model.training = mode
    except:
        pass
        
    # Limit visited set size to avoid memory leaks in large models
    if len(visited) > 10000:
        logger.warning("Too many modules in model, truncating traversal")
        return
        
    for name, child in model.named_children():
        if id(child) not in visited:
            safe_set_training_mode(child, mode, visited, depth + 1)


def get_memory_stats() -> Dict[str, float]:
    """Get comprehensive memory statistics"""
    stats = {
        "cpu_percent": psutil.cpu_percent(),
        "ram_used_gb": psutil.virtual_memory().used / (1024**3),
        "ram_available_gb": psutil.virtual_memory().available / (1024**3),
        "ram_percent": psutil.virtual_memory().percent
    }
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            stats.update({
                f"gpu_{i}_allocated_mb": torch.cuda.memory_allocated(i) / 1024**2,
                f"gpu_{i}_reserved_mb": torch.cuda.memory_reserved(i) / 1024**2,
                f"gpu_{i}_total_mb": torch.cuda.get_device_properties(i).total_memory / 1024**2,
                f"gpu_{i}_percent": (torch.cuda.memory_allocated(i) / torch.cuda.get_device_properties(i).total_memory) * 100
            })
    
    return stats


def log_memory_stats(prefix: str = ""):
    """Log current memory statistics"""
    stats = get_memory_stats()
    logger.info(f"{prefix} Memory Stats:")
    for key, value in stats.items():
        if 'percent' in key:
            logger.info(f"  {key}: {value:.1f}%")
        elif 'gb' in key or 'mb' in key:
            logger.info(f"  {key}: {value:.2f}")
        else:
            logger.info(f"  {key}: {value}")