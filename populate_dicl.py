#!/usr/bin/env python3
"""Simple script to populate the dICL database."""

import os
import sys

# Add the src directory to the Python path
sys.path.insert(0, 'src')

# Import the dICL system
from mmgraphrag_odsc_west_2025.lib.dicl_system import DICLSystem

def main():
    """Populate the dICL database with seed data."""
    print("Populating LanceDB database with dICL examples from seed data...")
    print("This will load examples from seed_data.yml for structured output, pattern learning, and voice & style.")
    print("Make sure Ollama is running with the nomic-embed-text:v1.5 model before proceeding.")
    print()
    
    try:
        system = DICLSystem()
        system.populate_database('dicl_examples', 'seed_data.yml')
        print('Database populated successfully!')
        print('Database location: dicl_examples')
        print('Loaded examples from: seed_data.yml')
    except Exception as e:
        print(f'Error populating database: {e}')
        sys.exit(1)

if __name__ == "__main__":
    main()
