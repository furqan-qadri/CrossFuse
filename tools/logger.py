#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
Simple Training Logger for CrossFuse
Saves training logs to text files for later analysis and plotting
"""

import os
import time
from datetime import datetime

class TrainingLogger:
    def __init__(self, log_dir="./logs", model_name="model"):
        """
        Initialize training logger
        
        Args:
            log_dir: Directory to save log files
            model_name: Name of the model for log file naming
        """
        self.log_dir = log_dir
        self.model_name = model_name
        self.log_file = None
        self.start_time = None
        
        # Create log directory
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"{model_name}_training_{timestamp}.log"
        self.log_file_path = os.path.join(log_dir, log_filename)
        
        # Open log file
        self.log_file = open(self.log_file_path, 'w')
        
        # Write header
        self.log_file.write(f"CrossFuse Training Log - {model_name}\n")
        self.log_file.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log_file.write("=" * 80 + "\n\n")
        
        print(f"Training logger initialized. Log file: {self.log_file_path}")
    
    def log_epoch_start(self, epoch, total_epochs):
        """Log start of an epoch"""
        self.start_time = time.time()
        epoch_msg = f"Epoch {epoch}/{total_epochs} started at {datetime.now().strftime('%H:%M:%S')}"
        self.log_file.write(f"\n{epoch_msg}\n")
        self.log_file.write("-" * 60 + "\n")
        self.log_file.flush()
        print(epoch_msg)
    
    def log_batch(self, epoch, total_epochs, batch, total_batches, losses, lr=None):
        """
        Log batch training information
        
        Args:
            epoch: Current epoch number
            total_epochs: Total number of epochs
            batch: Current batch number
            total_batches: Total batches per epoch
            losses: Dictionary of loss values
            lr: Learning rate (optional)
        """
        # Format losses
        loss_str = " - ".join([f"{k}: {v:.6f}" for k, v in losses.items()])
        
        # Create log message
        timestamp = datetime.now().strftime("%H:%M:%S")
        batch_msg = f"{timestamp} - Epoch {epoch}/{total_epochs} - Batch {batch}/{total_batches}"
        if lr is not None:
            batch_msg += f" - lr:{lr:.6f}"
        batch_msg += f" - {loss_str}"
        
        # Write to log file
        self.log_file.write(batch_msg + "\n")
        self.log_file.flush()
        
        # Print to console
        print(batch_msg)
    
    def log_epoch_end(self, epoch, total_epochs, epoch_losses):
        """Log end of an epoch with summary"""
        if self.start_time:
            epoch_time = time.time() - self.start_time
            time_msg = f"Epoch {epoch} completed in {epoch_time:.2f} seconds"
        else:
            time_msg = f"Epoch {epoch} completed"
        
        # Format epoch losses
        loss_summary = " - ".join([f"{k}: {v:.6f}" for k, v in epoch_losses.items()])
        
        epoch_end_msg = f"\n{time_msg}\nEpoch {epoch} Summary: {loss_summary}\n"
        epoch_end_msg += "-" * 60 + "\n"
        
        self.log_file.write(epoch_end_msg)
        self.log_file.flush()
        print(epoch_end_msg)
    
    def log_training_end(self, total_training_time):
        """Log end of training"""
        end_msg = f"\nTraining completed in {total_training_time:.2f} seconds\n"
        end_msg += f"Log saved to: {self.log_file_path}\n"
        end_msg += "=" * 80 + "\n"
        
        self.log_file.write(end_msg)
        self.log_file.flush()
        print(end_msg)
    
    def close(self):
        """Close the log file"""
        if self.log_file:
            self.log_file.close()
            print(f"Training log saved to: {self.log_file_path}")
    
    def __del__(self):
        """Destructor to ensure file is closed"""
        if hasattr(self, 'log_file') and self.log_file:
            self.log_file.close()

# Convenience function for quick logging
def log_training_step(logger, epoch, total_epochs, batch, total_batches, losses, lr=None):
    """Quick function to log a training step"""
    if logger:
        logger.log_batch(epoch, total_epochs, batch, total_batches, losses, lr) 