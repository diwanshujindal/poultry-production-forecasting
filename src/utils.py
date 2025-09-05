"""
Utility functions for the project
"""

import os

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
