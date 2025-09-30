import torch
import time
import json
import os
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import numpy as np
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Data class to store all model metrics"""
    total_parameters: int = 0
    trainable_parameters: int = 0
    model_size_mb: float = 0.0
    inference_memory_mb: float = 0.0
    inference_speed_tokens_per_sec: float = 0.0
    perplexity: float = 0.0
    size_reduction_percent: float = 0.0
    speed_change_percent: float = 0.0
    timestamp: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    def to_log_string(self) -> str:
        """Format metrics for logging"""
        return (
            f"Parameters: {self.total_parameters:,} | "
            f"Trainable Parameters: {self.trainable_parameters:,} | "
            f"Model Size (MB): {self.model_size_mb:.2f} | "
            f"Memory (Inference): {self.inference_memory_mb:.2f} MB | "
            f"Inference Speed: {self.inference_speed_tokens_per_sec:.2f} tokens/sec | "
            f"Perplexity: {self.perplexity:.2f} | "
            f"Size Reduction: {self.size_reduction_percent:.1f}% | "
            f"Speed Change: {self.speed_change_percent:+.1f}%"
        )


class MetricsTracker:
    """Track and log model metrics throughout training and inference"""
    
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.metrics_file = os.path.join(log_dir, "model_metrics.json")
        self.baseline_metrics: Optional[ModelMetrics] = None
        self.current_metrics = ModelMetrics()
        self.metrics_history: List[ModelMetrics] = []
        
        # Create metrics logger
        self.metrics_logger = self._setup_metrics_logger()
        
        # Load existing metrics if available
        self._load_metrics_history()
    
    def _setup_metrics_logger(self) -> logging.Logger:
        """Setup dedicated metrics logger"""
        metrics_logger = logging.getLogger('metrics_tracker')
        metrics_logger.setLevel(logging.INFO)
        
        # Create metrics log file
        os.makedirs(self.log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(self.log_dir, 'metrics.log'))
        fh.setLevel(logging.INFO)
        
        # Create formatter with the requested format
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        fh.setFormatter(formatter)
        metrics_logger.addHandler(fh)
        
        return metrics_logger
    
    def _load_metrics_history(self):
        """Load metrics history from file if exists"""
        if os.path.exists(self.metrics_file):
            try:
                with open(self.metrics_file, 'r') as f:
                    data = json.load(f)
                    self.metrics_history = [ModelMetrics(**item) for item in data.get('history', [])]
                    if data.get('baseline'):
                        self.baseline_metrics = ModelMetrics(**data['baseline'])
                logger.info(f"Loaded {len(self.metrics_history)} metrics records")
            except Exception as e:
                logger.warning(f"Failed to load metrics history: {e}")
    
    def save_metrics(self):
        """Save metrics to file"""
        try:
            data = {
                'baseline': self.baseline_metrics.to_dict() if self.baseline_metrics else None,
                'current': self.current_metrics.to_dict(),
                'history': [m.to_dict() for m in self.metrics_history]
            }
            with open(self.metrics_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
    
    def calculate_parameters(self, model: torch.nn.Module) -> tuple[int, int]:
        """Calculate total and trainable parameters"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total_params, trainable_params
    
    def calculate_model_size(self, model: torch.nn.Module, include_optimizer: bool = False) -> float:
        """Calculate model size in MB"""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024**2
        
        # Optionally include optimizer state
        if include_optimizer and hasattr(model, 'optimizer'):
            optimizer_size = sum(
                sum(v.nelement() * v.element_size() for v in state.values())
                for state in model.optimizer.state.values()
            )
            size_mb += optimizer_size / 1024**2
        
        return size_mb
    
    def measure_inference_speed(self, model: torch.nn.Module, tokenizer, 
                              test_text: str = "This is a test sentence for measuring inference speed.",
                              num_tokens: int = 50, num_runs: int = 5) -> float:
        """Measure inference speed in tokens per second"""
        model.eval()
        
        # Tokenize input
        inputs = tokenizer(test_text, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        total_time = 0
        total_tokens = 0
        
        with torch.no_grad():
            # Warmup
            model.generate(**inputs, max_new_tokens=10)
            
            # Actual measurement
            for _ in range(num_runs):
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.time()
                
                outputs = model.generate(**inputs, max_new_tokens=num_tokens)
                
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()
                
                total_time += (end_time - start_time)
                total_tokens += outputs.shape[1] - inputs['input_ids'].shape[1]
        
        tokens_per_second = total_tokens / total_time if total_time > 0 else 0
        return tokens_per_second
    
    def measure_inference_memory(self, model: torch.nn.Module, tokenizer,
                               test_text: str = "This is a test sentence.",
                               max_length: int = 512) -> float:
        """Measure memory usage during inference in MB"""
        if not torch.cuda.is_available():
            return 0.0
        
        # Clear cache
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Measure initial memory
        initial_memory = torch.cuda.memory_allocated() / 1024**2
        
        # Tokenize and run inference
        inputs = tokenizer(test_text, return_tensors="pt", max_length=max_length, truncation=True)
        inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100)
        
        torch.cuda.synchronize()
        
        # Measure final memory
        final_memory = torch.cuda.memory_allocated() / 1024**2
        
        # Return the difference
        return final_memory - initial_memory
    
    def calculate_perplexity(self, model: torch.nn.Module, dataloader: torch.utils.data.DataLoader,
                           max_batches: int = 50) -> float:
        """Calculate perplexity on a dataset"""
        model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= max_batches:
                    break
                
                # Move batch to device
                if torch.cuda.is_available():
                    batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                
                # Forward pass
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch.get('attention_mask'),
                    labels=batch.get('labels', batch['input_ids'])
                )
                
                # Accumulate loss
                total_loss += outputs.loss.item() * batch['input_ids'].numel()
                total_tokens += batch['input_ids'].numel()
        
        # Calculate perplexity
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return perplexity
    
    def update_metrics(self, model: torch.nn.Module, tokenizer, 
                      dataloader: Optional[torch.utils.data.DataLoader] = None,
                      is_baseline: bool = False) -> ModelMetrics:
        """Update all metrics for the model"""
        metrics = ModelMetrics()
        metrics.timestamp = datetime.now().isoformat()
        
        # Calculate parameters
        total_params, trainable_params = self.calculate_parameters(model)
        metrics.total_parameters = total_params
        metrics.trainable_parameters = trainable_params
        
        # Calculate model size
        metrics.model_size_mb = self.calculate_model_size(model)
        
        # Measure inference speed
        metrics.inference_speed_tokens_per_sec = self.measure_inference_speed(model, tokenizer)
        
        # Measure inference memory
        metrics.inference_memory_mb = self.measure_inference_memory(model, tokenizer)
        
        # Calculate perplexity if dataloader is provided
        if dataloader:
            metrics.perplexity = self.calculate_perplexity(model, dataloader)
        
        # Calculate relative changes if baseline exists
        if self.baseline_metrics and not is_baseline:
            # Size reduction
            if self.baseline_metrics.model_size_mb > 0:
                size_reduction = (self.baseline_metrics.model_size_mb - metrics.model_size_mb) / self.baseline_metrics.model_size_mb * 100
                metrics.size_reduction_percent = size_reduction
            
            # Speed change
            if self.baseline_metrics.inference_speed_tokens_per_sec > 0:
                speed_change = (metrics.inference_speed_tokens_per_sec - self.baseline_metrics.inference_speed_tokens_per_sec) / self.baseline_metrics.inference_speed_tokens_per_sec * 100
                metrics.speed_change_percent = speed_change
        
        # Update tracking
        if is_baseline:
            self.baseline_metrics = metrics
        else:
            self.current_metrics = metrics
            self.metrics_history.append(metrics)
        
        # Log metrics
        self.metrics_logger.info(metrics.to_log_string())
        
        # Save to file
        self.save_metrics()
        
        return metrics
    
    def log_comparison_table(self):
        """Log a comparison table of baseline vs current metrics"""
        if not self.baseline_metrics:
            logger.warning("No baseline metrics available for comparison")
            return
        
        header = "Metric | Baseline | Current | Change"
        separator = "-" * 50
        
        self.metrics_logger.info(separator)
        self.metrics_logger.info(header)
        self.metrics_logger.info(separator)
        
        # Parameters
        self.metrics_logger.info(
            f"Total Parameters | {self.baseline_metrics.total_parameters:,} | "
            f"{self.current_metrics.total_parameters:,} | "
            f"{self.current_metrics.total_parameters - self.baseline_metrics.total_parameters:,}"
        )
        
        # Trainable Parameters
        self.metrics_logger.info(
            f"Trainable Parameters | {self.baseline_metrics.trainable_parameters:,} | "
            f"{self.current_metrics.trainable_parameters:,} | "
            f"{self.current_metrics.trainable_parameters - self.baseline_metrics.trainable_parameters:,}"
        )
        
        # Model Size
        self.metrics_logger.info(
            f"Model Size (MB) | {self.baseline_metrics.model_size_mb:.2f} | "
            f"{self.current_metrics.model_size_mb:.2f} | "
            f"{self.current_metrics.size_reduction_percent:.1f}%"
        )
        
        # Inference Memory
        self.metrics_logger.info(
            f"Inference Memory (MB) | {self.baseline_metrics.inference_memory_mb:.2f} | "
            f"{self.current_metrics.inference_memory_mb:.2f} | "
            f"{(self.current_metrics.inference_memory_mb - self.baseline_metrics.inference_memory_mb):.2f}"
        )
        
        # Inference Speed
        self.metrics_logger.info(
            f"Inference Speed (tokens/sec) | {self.baseline_metrics.inference_speed_tokens_per_sec:.2f} | "
            f"{self.current_metrics.inference_speed_tokens_per_sec:.2f} | "
            f"{self.current_metrics.speed_change_percent:+.1f}%"
        )
        
        # Perplexity
        self.metrics_logger.info(
            f"Perplexity | {self.baseline_metrics.perplexity:.2f} | "
            f"{self.current_metrics.perplexity:.2f} | "
            f"{(self.current_metrics.perplexity - self.baseline_metrics.perplexity):.2f}"
        )
        
        self.metrics_logger.info(separator)