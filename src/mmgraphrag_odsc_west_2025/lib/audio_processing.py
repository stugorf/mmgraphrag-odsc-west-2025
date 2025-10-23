"""Audio processing utilities for generating embeddings from audio and text.

This module provides functionality for processing audio.

The main class AudioProcessing handles audio processing, embedding generation,
and TensorBoard visualization.

Example:
    processor = AudioProcessing()
    num_samples = 100     
    processor.process_audios(audio_dir="path/to/audio", num_samples=num_samples)

Attributes:
    open: Function to open audio files (soundfile.read)
    logger: Logger instance for this class
"""

# Standard library imports
import csv
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Third party imports
from tqdm import tqdm
import base64
import soundfile as sf
import numpy as np
from PIL import Image
from dotenv import load_dotenv

# Note: TensorFlow removed as it's not needed for audio processing

# Local imports
from mmgraphrag_odsc_west_2025.config import config
from mmgraphrag_odsc_west_2025.utils.logger import setup_logger


# Load environment variables from .env file
# Look for .env file in the src directory relative to this file
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)

# Set OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
else:
    # For demo purposes, set a placeholder if not found
    print("Warning: OPENAI_API_KEY not found in .env file. Some audio processing features may not work.")
    os.environ["OPENAI_API_KEY"] = "demo_key"

# Set BAML Logging [error, warn, info, debug, trace, off]
os.environ["BAML_LOG"] = config.BAML_LOG

# BAML imports
import mmgraphrag_odsc_west_2025.baml_client as baml



class AudioProcessing:
    """Class for audio processing utilities."""

    def __init__(self):
        self.open = sf.read
        self.logger = setup_logger(__name__)

    # Function to load the audio and convert it to base64 format
    def audio_to_base64(self, audio_path):
        """Convert an audio file to base64 format.
        
        Args:
            audio_path (str): Path to the audio file to convert
            
        Returns:
            str: Base64 encoded string representation of the audio, or None if conversion fails
            
        Raises:
            Exception: If there is an error reading or encoding the audio file
        """
        try:
            with open(audio_path, "rb") as audio_file:
                return base64.b64encode(audio_file.read()).decode('utf-8')
        except Exception as e:
            self.logger.error(f"Error processing audio: {e}")
            return None

    def process_single_audio(self, audio_path: Path, audio_id: int):
        """Process a single audio file and return its analysis and metadata.
        
        This function takes an audio file, generates an analysis using BAML's AnalyzeAudio,
        and returns the results as a dictionary containing the audio name and analysis.

        Args:
            audio_path (Path): Path object pointing to the audio file
            audio_id (int): Unique identifier for the audio

        Returns:
            dict: Dictionary containing:
                - name (str): Name of the audio file
                - analysis (str): Generated analysis for the audio
                
        Raises:
            Exception: If there is an error processing the audio, generating analysis,
            or creating embeddings
        """
        try:
            self.logger.debug(f"Processing audio {audio_path}")
            
            # Open audio and convert to base64
            audio, sr = self.open(audio_path)
            audio_base64 = self.audio_to_base64(audio_path)

            # Construct a BAML Image Object
            audio_object = {
                "base64":audio_base64,
                "media_type": "audio/mp3"
            }

            # Extract caption from image using BAML
            response = baml.b.AnalyzeAudio(audio_object)
            analysis = response.analysis
            
            return {
                "name": audio_path.name,
                "analysis": analysis
            }

        except Exception as e:
            self.logger.error(f"Error processing {audio_path}: {e}", exc_info=True)
            return None


    def process_audios(self, audio_dir: str, num_samples: int):
        """
        Process audio files and extract analysis using BAML.

        This method processes a batch of audio files by extracting analysis using BAML.
        It samples audio files from the provided directory and processes them for analysis.

        Args:
            audio_dir (str): Directory containing the audio files to process
            num_samples (int): Number of random audio files to sample and process

        Returns:
            int: Number of successfully processed audio files

        Raises:
            ValueError: If no audio files are successfully processed
            Exception: If there are errors processing individual audio files
        """
        try:
            # Load audio files
            audio_files = list(Path(audio_dir).glob('*.mp3'))
            sampled_files = random.sample(audio_files, min(num_samples, len(audio_files)))

            # Process audio files and collect successful entries
            successful_entries = []
            
            with tqdm(total=len(sampled_files), desc="Processing audio") as pbar:
                for audio_path in sampled_files:
                    try:
                        # Process single audio file
                        result = self.process_single_audio(
                            audio_path, 
                            int(audio_path.stem.split('_')[-1])
                        )
                        
                        if result:
                            successful_entries.append({
                                'path': str(audio_path),
                                'analysis': result['analysis']
                            })
                                
                    except Exception as e:
                        self.logger.error(f"Error processing {audio_path}: {e}")
                    finally:
                        pbar.update(1)

            if not successful_entries:
                raise ValueError("No audio files were successfully processed")

            self.logger.info(f"Successfully processed {len(successful_entries)} audio files")
            return len(successful_entries)

        except Exception as e:
            self.logger.error(f"Error in process_audios: {e}")
            raise

