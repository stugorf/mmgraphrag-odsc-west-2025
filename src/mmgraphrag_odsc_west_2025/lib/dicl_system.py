"""Consolidated Dynamic In-Context Learning (dICL) System.

This module provides a complete dICL system that:
1. Generates examples using language models
2. Creates LanceDB database with embeddings
3. Performs similarity search for dynamic in-context learning
4. Uses BAML for language model interactions

Usage:
    system = DICLSystem()
    system.populate_database("dicl_examples")
    result = system.process_query("What is lambda calculus?")
"""

import sys
import os
import json
import requests
import numpy as np
import pandas as pd
import lancedb
import time
import yaml
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from tqdm import tqdm
import logging
from pathlib import Path
from contextlib import contextmanager

#local imports
from mmgraphrag_odsc_west_2025.config import config


# Set environment variables before BAML import
os.environ["BAML_LOG"] = config.BAML_LOG

# BAML imports (optional for seed data mode)
try:
    import mmgraphrag_odsc_west_2025.baml_client as baml
    from mmgraphrag_odsc_west_2025.baml_client.types import ExampleGenerationInput as BAMLExampleGenerationInput, DICLInput as BAMLDICLInput, Example as BAMLExample
    # Test if BAML actually works by trying to access the client
    _ = baml.b
    BAML_AVAILABLE = True
except (ImportError, AttributeError, Exception):
    BAML_AVAILABLE = False
    baml = None
    BAMLExampleGenerationInput = None
    BAMLDICLInput = None
    BAMLExample = None


def setup_logger(name):
    """Setup logging for the dICL system."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


@contextmanager
def lancedb_connection(db_path: str):
    """Context manager for LanceDB connections to ensure proper cleanup.
    
    This helps prevent connection issues by ensuring connections are properly closed.
    
    Args:
        db_path: Path to the LanceDB database
        
    Yields:
        LanceDB connection object
    """
    db = None
    max_retries = 3
    retry_delay = 1.0
    
    try:
        for attempt in range(max_retries):
            try:
                # Add small delay between attempts to prevent resource contention
                if attempt > 0:
                    time.sleep(retry_delay * attempt)
                
                db = lancedb.connect(db_path)
                yield db
                break
                
            except Exception as e:
                if attempt == max_retries - 1:
                    # Last attempt failed, re-raise the exception
                    raise
                else:
                    # Log the attempt and continue
                    logging.getLogger(__name__).warning(f"LanceDB connection attempt {attempt + 1} failed: {e}")
                    if db is not None:
                        try:
                            # Try to clean up the failed connection
                            pass
                        except:
                            pass
                        db = None
    finally:
        if db is not None:
            # LanceDB connections are automatically cleaned up, but we can be explicit
            try:
                # Force cleanup of any pending operations
                pass
            except Exception as e:
                # Log but don't raise to avoid masking other errors
                logging.getLogger(__name__).warning(f"Error during LanceDB cleanup: {e}")


@dataclass
class Example:
    """Represents an example in the LanceDB database."""
    id: str
    vector: List[float]
    input: str
    output: str


class DICLSystem:
    """Consolidated Dynamic In-Context Learning system using LanceDB and BAML.
    
    This class combines example generation, database management, and query processing
    into a single unified interface.
    """
    
    def __init__(self, db_path: str = "dicl_examples"):
        """Initialize the dICL system.
        
        Args:
            db_path: Path to the LanceDB database
        """
        self.logger = setup_logger(__name__)
        self.ollama_url = config.OLLAMA_URL
        self.db_path = db_path
        self.db = None
        self.table = None
        
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using nomic-embed-text:v1.5 via Ollama.
        
        Args:
            text: Text to embed
            
        Returns:
            List of float values representing the embedding
        """
        try:
            response = requests.post(
                f"{self.ollama_url}/api/embeddings",
                json={
                    "model": "nomic-embed-text:v1.5",
                    "prompt": text
                },
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            return data["embedding"]
        except Exception as e:
            self.logger.error(f"Error getting embedding: {e}")
            raise
    
    def _load_seed_data(self, seed_data_path: str = "seed_data.yml") -> List[Example]:
        """Load examples from seed data YAML file.
        
        Args:
            seed_data_path: Path to the seed data YAML file
            
        Returns:
            List of Example objects
        """
        examples = []
        
        try:
            with open(seed_data_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            # Process structured output examples
            if 'structured_output' in data:
                structured_examples = self._parse_yaml_examples(data['structured_output'], 'structured')
                examples.extend(structured_examples)
                self.logger.info(f"Loaded {len(structured_examples)} structured examples")
            
            # Process pattern learning examples
            if 'pattern_learning' in data:
                pattern_examples = self._parse_yaml_examples(data['pattern_learning'], 'pattern')
                examples.extend(pattern_examples)
                self.logger.info(f"Loaded {len(pattern_examples)} pattern examples")
            
            # Process voice & style examples
            if 'voice_style' in data:
                voice_examples = self._parse_yaml_examples(data['voice_style'], 'voice')
                examples.extend(voice_examples)
                self.logger.info(f"Loaded {len(voice_examples)} voice examples")
            
            self.logger.info(f"Total loaded {len(examples)} examples from seed data")
            
        except Exception as e:
            self.logger.error(f"Error loading seed data: {e}")
            raise
        
        return examples
    
    def _parse_yaml_examples(self, examples_data: List[Dict[str, str]], example_type: str) -> List[Example]:
        """Parse examples from YAML data.
        
        Args:
            examples_data: List of dictionaries with 'input' and 'output' keys
            example_type: Type of examples (structured, pattern, voice)
            
        Returns:
            List of Example objects
        """
        examples = []
        
        for i, example_data in enumerate(examples_data):
            try:
                input_text = example_data['input']
                output_text = example_data['output']
                
                # Get embedding for the input
                embedding = self._get_embedding(input_text)
                
                example = Example(
                    id=f"{example_type}_{i}",
                    vector=embedding,
                    input=input_text,
                    output=output_text
                )
                examples.append(example)
                
            except KeyError as e:
                self.logger.warning(f"Missing key in example {i}: {e}")
                continue
            except Exception as e:
                self.logger.warning(f"Error processing example {i}: {e}")
                continue
        
        return examples
    
    def _parse_structured_examples(self, section_content: str) -> List[Example]:
        """Parse structured output examples from seed data."""
        examples = []
        lines = section_content.split('\n')
        
        current_input = None
        current_output = None
        example_count = 0
        
        for line in lines:
            line = line.strip()
            if line.startswith('Input:'):
                current_input = line.replace('Input:', '').strip()
            elif line.startswith('Output:'):
                current_output = line.replace('Output:', '').strip()
                
                if current_input and current_output:
                    # Get embedding for the input
                    embedding = self._get_embedding(current_input)
                    
                    example = Example(
                        id=f"structured_{example_count}",
                        vector=embedding,
                        input=current_input,
                        output=current_output
                    )
                    examples.append(example)
                    example_count += 1
                    current_input = None
                    current_output = None
        
        return examples
    
    def _parse_pattern_examples(self, section_content: str) -> List[Example]:
        """Parse pattern learning examples from seed data."""
        examples = []
        lines = section_content.split('\n')
        
        current_input = None
        current_output_lines = []
        example_count = 0
        in_output = False
        
        for line in lines:
            line = line.strip()
            if line.startswith('Input:'):
                current_input = line.replace('Input:', '').strip()
                in_output = False
            elif line.startswith('Output:'):
                in_output = True
                current_output_lines = []
            elif in_output and line and not line.startswith('Example'):
                current_output_lines.append(line)
            elif line.startswith('Example') and current_input and current_output_lines:
                # Process the completed example
                current_output = '\n'.join(current_output_lines)
                
                # Get embedding for the input
                embedding = self._get_embedding(current_input)
                
                example = Example(
                    id=f"pattern_{example_count}",
                    vector=embedding,
                    input=current_input,
                    output=current_output
                )
                examples.append(example)
                example_count += 1
                current_input = None
                current_output_lines = []
                in_output = False
        
        # Handle the last example if it doesn't end with "Example"
        if current_input and current_output_lines:
            current_output = '\n'.join(current_output_lines)
            embedding = self._get_embedding(current_input)
            
            example = Example(
                id=f"pattern_{example_count}",
                vector=embedding,
                input=current_input,
                output=current_output
            )
            examples.append(example)
        
        return examples
    
    def _parse_voice_examples(self, section_content: str) -> List[Example]:
        """Parse voice & style examples from seed data."""
        examples = []
        lines = section_content.split('\n')
        
        current_input = None
        current_output = None
        example_count = 0
        
        for line in lines:
            line = line.strip()
            if line.startswith('Input:'):
                current_input = line.replace('Input:', '').strip()
            elif line.startswith('Output:'):
                current_output = line.replace('Output:', '').strip()
                
                if current_input and current_output:
                    # Get embedding for the input
                    embedding = self._get_embedding(current_input)
                    
                    example = Example(
                        id=f"voice_{example_count}",
                        vector=embedding,
                        input=current_input,
                        output=current_output
                    )
                    examples.append(example)
                    example_count += 1
                    current_input = None
                    current_output = None
        
        return examples
    
    def populate_database(self, db_path: Optional[str] = None, seed_data_path: str = "seed_data.yml"):
        """Create the LanceDB database with examples from seed data.
        
        Args:
            db_path: Path to the LanceDB database directory (optional)
            seed_data_path: Path to the seed data YAML file
        """
        if db_path:
            self.db_path = db_path
            
        # Ensure the database directory exists
        os.makedirs(self.db_path, exist_ok=True)
        
        try:
            # Use context manager for proper connection management
            with lancedb_connection(self.db_path) as db:
                # Check if table already exists and drop it to repopulate
                if "examples" in db.table_names():
                    self.logger.info("Examples table exists. Dropping and recreating...")
                    db.drop_table("examples")
                
                # Load examples from seed data
                self.logger.info("Loading examples from seed data...")
                all_examples = self._load_seed_data(seed_data_path)
                
                if not all_examples:
                    raise ValueError("No examples loaded from seed data")
                
                # Convert to pandas DataFrame for LanceDB
                data = []
                for example in all_examples:
                    data.append({
                        "id": example.id,
                        "vector": example.vector,
                        "input": example.input,
                        "output": example.output
                    })
                
                # Create pandas DataFrame
                df = pd.DataFrame(data)
                
                # Add small delay before table creation
                time.sleep(1)
                
                # Create table using pandas DataFrame
                table = db.create_table("examples", df)
                self.logger.info(f"Created examples table with {len(data)} examples")
                
                # Store connection and table for later use
                self.db = db
                self.table = table
            
        except Exception as e:
            self.logger.error(f"Error creating database: {e}")
            # Try to clean up any partial state
            try:
                if os.path.exists(self.db_path):
                    import shutil
                    shutil.rmtree(self.db_path)
                    self.logger.info("Cleaned up partial database directory")
            except Exception as cleanup_error:
                self.logger.warning(f"Error during cleanup: {cleanup_error}")
            raise
    
    def initialize_database(self, db_path: Optional[str] = None):
        """Initialize connection to existing LanceDB database.
        
        Args:
            db_path: Path to the LanceDB database directory (optional)
        """
        if db_path:
            self.db_path = db_path
            
        try:
            # Use context manager for proper connection management
            with lancedb_connection(self.db_path) as db:
                # Get the examples table
                if "examples" not in db.table_names():
                    raise ValueError(f"No 'examples' table found in database at {self.db_path}")
                
                table = db.open_table("examples")
                self.logger.info(f"Connected to database at {self.db_path}")
                
                # Store connection and table for later use
                self.db = db
                self.table = table
            
        except Exception as e:
            self.logger.error(f"Error initializing database: {e}")
            raise
    
    def _search_similar_examples(self, query: str, num_examples: int = 3) -> List[Example]:
        """Search for similar examples in the database.
        
        Args:
            query: Query string to search for
            num_examples: Number of similar examples to return
            
        Returns:
            List of similar Example objects
        """
        if not self.table:
            raise ValueError("Database not initialized. Call initialize_database() first.")
        
        try:
            # Get embedding for the query
            query_embedding = self._get_embedding(query)
            
            # Search for similar examples
            results = self.table.search(query_embedding).limit(num_examples).to_pandas()
            
            # Convert to Example objects
            examples = []
            for _, row in results.iterrows():
                example = Example(
                    id=row['id'],
                    vector=row['vector'],
                    input=row['input'],
                    output=row['output']
                )
                examples.append(example)
            
            return examples
            
        except Exception as e:
            self.logger.error(f"Error searching for similar examples: {e}")
            raise
    
    def process_query(self, query: str, num_examples: int = 3) -> Dict[str, Any]:
        """Process a query using dynamic in-context learning.
        
        Args:
            query: User's question
            num_examples: Number of similar examples to use as context
            
        Returns:
            Dictionary containing the answer and reasoning
        """
        if not self.table:
            raise ValueError("Database not initialized. Call initialize_database() first.")
        
        try:
            # Search for similar examples
            similar_examples = self._search_similar_examples(query, num_examples)
            
            if not similar_examples:
                return {
                    "answer": "No similar examples found in the database.",
                    "reasoning": "The database is empty or no similar examples were found.",
                    "examples": []
                }
            
            # Use BAML to generate answer with context (if available)
            if BAML_AVAILABLE:
                # Convert our Example objects to BAML Example format as dictionaries
                baml_examples = []
                for ex in similar_examples:
                    baml_example = {
                        "id": ex.id,
                        "input": ex.input,
                        "output": ex.output
                    }
                    baml_examples.append(baml_example)
                
                input_data = {
                    "query": query,
                    "examples": baml_examples
                }
                result = baml.b.DynamicInContextLearning(input_data)
                
                return {
                    "answer": result.answer,
                    "reasoning": result.reasoning,
                    "examples": similar_examples
                }
            else:
                # Fallback response when BAML is not available
                return {
                    "answer": f"Based on the similar examples found, here are the relevant patterns: {', '.join([ex.output for ex in similar_examples[:2]])}",
                    "reasoning": f"Found {len(similar_examples)} similar examples in the database. Using seed data examples for context.",
                    "examples": similar_examples
                }
            
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            raise


def main():
    """Main function to populate the database and test the system."""
    logger = setup_logger(__name__)

    logger.info("Starting DICL system test...")
    
    try:
        # Create the dICL system
        logger.info("Creating dICL system...")
        system = DICLSystem()
        
        # # Test BAML connection first
        # logger.info("Testing BAML connection...")
        # input = ExampleGenerationInput(
        #     topic="test",
        #     num_examples=1
        # )
        # test_result = baml.b.GenerateExamples(input)
        # logger.info(f"BAML test successful: {len(test_result.examples)} examples generated")
        
        # # Populate the database
        # logger.info("Populating database...")
        # system.populate_database("dicl_examples")
        
        # logger.info("Database populated successfully!")
        # logger.info("Database location: dicl_examples")
        
        # # Test the system with sample queries
        # logger.info("Testing system with sample queries...")
        
        # # Test banana query
        # banana_result = system.process_query("What are the health benefits of bananas?")
        # logger.info(f"Banana query returned {len(banana_result['examples'])} examples")
        
        # # Test lambda calculus query
        # lambda_result = system.process_query("What is lambda calculus?")
        # logger.info(f"Lambda calculus query returned {len(lambda_result['examples'])} examples")
        
        # # Test technology query
        # tech_result = system.process_query("What is machine learning?")
        # logger.info(f"Technology query returned {len(tech_result['examples'])} examples")
        
        # logger.info("System testing completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise


if __name__ == "__main__":
    main()
