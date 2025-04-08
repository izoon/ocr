import time
import psutil
import os
from typing import Dict, Any
from contextlib import contextmanager
from api.core.logging import logger

@contextmanager
def measure_performance(operation_name: str):
    """Context manager for measuring operation performance."""
    start_time = time.time()
    start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
    
    try:
        yield
    finally:
        end_time = time.time()
        end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
        
        duration = end_time - start_time
        memory_used = end_memory - start_memory
        
        logger.info(f"Performance metrics for {operation_name}:")
        logger.info(f"  Duration: {duration:.2f} seconds")
        logger.info(f"  Memory used: {memory_used:.2f} MB")
        
        return {
            "time": duration,
            "memory_used": memory_used
        }

def get_system_metrics() -> Dict[str, Any]:
    """Get current system metrics."""
    process = psutil.Process(os.getpid())
    return {
        "cpu_percent": process.cpu_percent(),
        "memory_percent": process.memory_percent(),
        "memory_info": {
            "rss": process.memory_info().rss / 1024 / 1024,  # MB
            "vms": process.memory_info().vms / 1024 / 1024   # MB
        },
        "num_threads": process.num_threads(),
        "num_fds": process.num_fds()
    } 