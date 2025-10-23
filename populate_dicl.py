#!/usr/bin/env python3
"""Simple script to populate the dICL database."""

import os
import sys

# Add the src directory to the Python path
sys.path.insert(0, 'src')

from mmgraphrag_odsc_west_2025.lib.dicl_system import DICLSystem

def main():
    """Populate the dICL database."""
    print("Populating LanceDB database with dICL examples...")
    print("This will generate examples for bananas, technology, and functional programming topics.")
    print("Make sure Ollama is running with the phi4 model before proceeding.")
    print()
    
    try:
        system = DICLSystem()
        system.populate_database('dicl_examples')
        print('Database populated successfully!')
        print('Database location: dicl_examples')
    except Exception as e:
        print(f'Error populating database: {e}')
        sys.exit(1)

if __name__ == "__main__":
    main()
