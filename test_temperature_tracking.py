#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple test script to verify temperature tracking functionality
"""

import torch
import torch.nn as nn
import numpy as np
from temperature_tracker import TemperatureTracker

# Mock cross-attention module for testing
class MockCrossAttention(nn.Module):
    def __init__(self, n_heads=16):
        super().__init__()
        self.cross = True
        self.temperatures = nn.Parameter(torch.ones(n_heads))
    
    def forward(self, x):
        return x

# Mock model with cross-attention modules
class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_attn1 = MockCrossAttention(16)
        self.cross_attn2 = MockCrossAttention(16)
        
    def forward(self, x):
        return x

def test_temperature_tracking():
    """Test the temperature tracking system"""
    print("🔥 Testing Temperature Tracking System...")
    
    # Create mock model
    model = MockModel()
    
    # Initialize tracker
    tracker = TemperatureTracker(save_dir="test_temp_logs")
    
    # Simulate training epochs with evolving temperatures
    for epoch in range(1, 6):
        # Simulate temperature evolution (some heads specializing)
        with torch.no_grad():
            # Make some heads go sharp (< 0.7)
            model.cross_attn1.temperatures[0] = 0.5 + epoch * 0.02
            model.cross_attn1.temperatures[1] = 0.6 - epoch * 0.05
            
            # Make some heads go broad (> 1.3)
            model.cross_attn1.temperatures[14] = 1.2 + epoch * 0.08
            model.cross_attn1.temperatures[15] = 1.1 + epoch * 0.12
            
            # Keep others balanced with some variation
            for i in range(2, 14):
                model.cross_attn1.temperatures[i] = 0.9 + np.random.normal(0, 0.1)
        
        # Track temperatures
        temps = tracker.collect_temperatures(model, epoch)
        
        if temps is not None:
            print(f"Epoch {epoch}: Variance = {np.var(temps):.4f}")
    
    # Save final summary
    tracker.save_final_summary()
    
    print("✅ Temperature tracking test completed!")
    print("Check 'test_temp_logs/' directory for generated files.")

if __name__ == "__main__":
    test_temperature_tracking()
