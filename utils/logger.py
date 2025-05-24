"""
Simple logger for AdaManip RGBD collection system.
"""

import os
import sys
from datetime import datetime
from pathlib import Path


class Logger:
    """Simple logger for RGBD data collection."""
    
    def __init__(self, log_dir: str = "./logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"rgbd_collection_{timestamp}.log"
        
        # Initialize log file
        self._write_log("INFO", "Logger initialized")
    
    def _write_log(self, level: str, message: str):
        """Write message to log file and console."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] [{level}] {message}"
        
        # Write to console
        print(log_message)
        
        # Write to file
        try:
            with open(self.log_file, 'a') as f:
                f.write(log_message + "\n")
        except:
            # If file writing fails, just continue
            pass
    
    def info(self, message: str):
        """Log info message."""
        self._write_log("INFO", message)
    
    def warning(self, message: str):
        """Log warning message."""
        self._write_log("WARNING", message)
    
    def error(self, message: str):
        """Log error message."""
        self._write_log("ERROR", message)
    
    def debug(self, message: str):
        """Log debug message."""
        self._write_log("DEBUG", message)
    
    def log(self, message: str, level: str = "INFO"):
        """Generic log method."""
        self._write_log(level.upper(), message) 