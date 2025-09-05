"""
Utility functions for the project
"""

import os
import json

def setup_directories():
    """
    Create necessary directories for the project
    """
    directories = ['data', 'notebooks', 'src', 'models', 'results']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        else:
            print(f"Directory already exists: {directory}")

def load_config(config_path='config.json'):
    """
    Load configuration from JSON file
    
    Parameters:
    config_path (str): Path to config file
    
    Returns:
    dict: Configuration parameters
    """
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        # Return default config if file doesn't exist
        return {
            "input_size": 7,
            "output_size": 2,
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "epochs": 50,
            "batch_size": 32
        }
