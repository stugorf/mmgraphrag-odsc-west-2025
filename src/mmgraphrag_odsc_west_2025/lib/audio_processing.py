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
import tensorflow as tf
from tensorboard.plugins import projector
from PIL import Image
from dotenv import load_dotenv

# Local imports
from mmgraphrag_odsc_west_2025.config import config
from mmgraphrag_odsc_west_2025.utils.logger import setup_logger


# Load environment variables from .env file
load_dotenv()

# Set OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
else:
    raise ValueError("OPENAI_API_KEY is not set in .env file")

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
        Process audio files, extract analysis, and prepare TensorBoard files.

        This method processes a batch of audio files by extracting analysis using BAML,
        and prepares files needed for TensorBoard visualization. It samples audio files from the
        provided directory, processes them using BAML for analysis, and generates
        sprite images and metadata files for TensorBoard.

        Args:
            audio_dir (str): Directory containing the audio files to process
            num_samples (int): Number of random audio files to sample and process

        Raises:
            ValueError: If no audio files are successfully processed or sprite image creation fails
            Exception: If there are errors processing individual audio files
        """
        try:
            # Define paths for tensorboard
            tensorboard_dir = config.TENSORBOARD_LOG_DIR
            vectors_path = os.path.join(tensorboard_dir, 'vectors.tsv')
            metadata_path = os.path.join(tensorboard_dir, 'metadata.tsv')

            # Create tensorboard directory
            os.makedirs(tensorboard_dir, exist_ok=True)
            self.logger.info(f"TensorBoard directory ready at {tensorboard_dir}")

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

            # Create sprite image (for audio visualization)
            sprite_paths = [entry['path'] for entry in successful_entries]
            sprite_image, final_processed_paths = self.create_sprite_image(sprite_paths)
            if not sprite_image or not final_processed_paths:
                raise ValueError("Failed to create sprite image")

            # Create mapping and sort entries
            sprite_order = {path: idx for idx, path in enumerate(final_processed_paths)}
            successful_entries.sort(key=lambda x: sprite_order[x['path']])
            
            # Build synchronized lists
            final_vectors = []
            final_metadata = []
            
            for entry in successful_entries:
                # For audio, we'll create a simple vector representation
                # This could be enhanced with actual audio embeddings
                audio_vector = [0.0] * 512  # Placeholder vector
                final_vectors.append(audio_vector)
                metadata_row = [
                    os.path.basename(entry['path']),  # Audio filename
                    entry['analysis'],  # Analysis
                ]
                final_metadata.append(metadata_row)

            # Write files
            with open(vectors_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter='\t')
                vectors_written = 0
                for vector in final_vectors:
                    writer.writerow(vector)
                    vectors_written += 1
                
                self.logger.info(f"Wrote {vectors_written} vectors out of {len(successful_entries)} successful entries")
                if vectors_written != len(successful_entries):
                    raise ValueError(f"Mismatch in number of vectors written ({vectors_written}) vs successful entries ({len(successful_entries)})")

            with open(metadata_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow(['Audio', 'Analysis'])
                for row in final_metadata:
                    cleaned_analysis = ' '.join(row[1].replace('\n', ' ').split())
                    writer.writerow([row[0], cleaned_analysis])

            # Save sprite image
            sprite_image.save(os.path.join(tensorboard_dir, 'sprite.png'))

            # Save checkpoint properly
            checkpoint_path = os.path.join(tensorboard_dir, 'model.ckpt')
            
            # Create a new graph and session to ensure clean state
            graph = tf.Graph()
            with graph.as_default():
                # Convert to numpy array and ensure correct shape
                embeddings_array = np.array(final_vectors)
                
                # Create the Variable with explicit shape
                embedding_var = tf.Variable(
                    embeddings_array,
                    trainable=False,
                    name='embedding'
                )
                
                # Initialize saver
                saver = tf.compat.v1.train.Saver([embedding_var])
                
                # Create session and initialize variables
                with tf.compat.v1.Session() as sess:
                    sess.run(tf.compat.v1.global_variables_initializer())
                    # Save the checkpoint
                    saver.save(sess, checkpoint_path)

            # Configure projector
            tensorboard_config = projector.ProjectorConfig()
            embedding = tensorboard_config.embeddings.add()
            embedding.tensor_name = "embedding"  # Must match the variable name above
            embedding.metadata_path = os.path.basename(metadata_path)
            embedding.sprite.image_path = 'sprite.png'
            embedding.sprite.single_image_dim.extend([100, 100])
            
            projector.visualize_embeddings(tensorboard_dir, tensorboard_config)
            
            return len(successful_entries)

        except Exception as e:
            self.logger.error(f"Error in process_audios: {e}")
            raise


    def create_sprite_image(self, images_paths: List[str], sprite_size: int = config.SPRITE_SIZE) -> Tuple[Image.Image | None, List[str]]:
        """Create a sprite image and return both the image and the order of processed paths.
        
        Creates a sprite image by combining multiple images into a single grid-like image,
        with each input image resized and placed on a white background. The sprite image
        is used for visualization in TensorBoard's Projector.

        Args:
            images_paths (List[str]): List of paths to images to include in sprite
            sprite_size (int, optional): Size in pixels for each image in the sprite. 
                Defaults to config.SPRITE_SIZE.

        Returns:
            Tuple[Image.Image, List[str]]: A tuple containing:
                - The generated sprite image as a PIL Image
                - List of successfully processed image paths in order of appearance
                
        Note:
            Images that fail to process are replaced with blank white squares in the 
            sprite but are not included in the returned list of successful paths.
        """
        if not images_paths:
            return None, []
        
        # Calculate grid dimensions
        num_images = len(images_paths)
        grid_size = int(np.ceil(np.sqrt(num_images)))
        
        # Create blank sprite image
        sprite = Image.new(config.IMAGE_FORMAT, 
                         (grid_size * sprite_size, grid_size * sprite_size), 
                         config.SPRITE_BG_COLOR)
        
        # Track successful placements and their order
        successful_paths = []
        
        for idx, img_path in enumerate(images_paths):
            try:
                # Open and resize image
                img = Image.open(img_path).convert(config.IMAGE_FORMAT)
                img.thumbnail((sprite_size, sprite_size), Image.Resampling.LANCZOS)
                
                # Create white background
                img_with_bg = Image.new(config.IMAGE_FORMAT, 
                                      (sprite_size, sprite_size), 
                                      config.SPRITE_BG_COLOR)
                offset = ((sprite_size - img.size[0]) // 2, (sprite_size - img.size[1]) // 2)
                img_with_bg.paste(img, offset)
                
                # Calculate position
                row = idx // grid_size
                col = idx % grid_size
                
                # Paste into sprite
                sprite.paste(img_with_bg, (col * sprite_size, row * sprite_size))
                successful_paths.append(img_path)
                
            except Exception as e:
                self.logger.error(f"Error processing image for sprite: {img_path} - {e}")
                # Create blank placeholder but don't add to successful paths
                img_with_bg = Image.new(config.IMAGE_FORMAT, (sprite_size, sprite_size), 'white')
                row = idx // grid_size
                col = idx % grid_size
                sprite.paste(img_with_bg, (col * sprite_size, row * sprite_size))
        
        return sprite, successful_paths