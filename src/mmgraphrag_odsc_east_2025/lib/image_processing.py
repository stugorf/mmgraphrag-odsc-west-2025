"""Image processing utilities for generating embeddings from images and text.

This module provides functionality for processing images and generating embeddings
from image-text pairs using CLIP (Contrastive Language-Image Pre-Training) models.
The embeddings can be used for multimodal similarity search and retrieval.

The main class ImageProcessing handles image processing, embedding generation,
and TensorBoard visualization.

Example:
    processor = ImageProcessing()
    num_samples = 100     
    processor.process_images(img_dir="path/to/images", num_samples=num_samples)

Attributes:
    open: Function to open images
    logger: Logger instance for this class
    embeddings: SemanticEmbeddings instance for embedding generation    
"""

# Standard library imports
import base64
import csv
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Third party imports
from PIL import Image
import numpy as np
import torch
from tensorboard.plugins import projector 
import tensorflow as tf
from tensorflow.keras.utils import img_to_array
from tqdm import tqdm

# Local imports
from mmgraphrag_odsc_aib2025.config import config
from mmgraphrag_odsc_aib2025.lib.embedding import SemanticEmbeddings
from mmgraphrag_odsc_aib2025.utils.logger import setup_logger



# Set environment variable for tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set BAML Logging [error, warn, info, debug, trace, off]
os.environ["BAML_LOG"] = config.BAML_LOG

# BAML imports
import mmgraphrag_odsc_aib2025.baml_client as baml



class ImageProcessing:
    """Class for image processing utilities."""

    def __init__(self):
        self.open = Image.open
        self.logger = setup_logger(__name__)
        self.embeddings = SemanticEmbeddings()

    # Function to load the image and convert it to base64 format
    def image_to_base64(self, image_path):
        """Convert an image file to base64 format.
        
        Args:
            image_path (str): Path to the image file to convert
            
        Returns:
            str: Base64 encoded string representation of the image, or None if conversion fails
            
        Raises:
            Exception: If there is an error reading or encoding the image file
        """
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            self.logger.error(f"Error processing image: {e}")
            return None

    def process_single_image(self, img_path: Path, img_id: int):
        """Process a single image and return its embedding and metadata.
        
        This function takes an image file, generates a caption using BAML's ExtractImageCaption,
        and creates a CLIP embedding combining both the image and caption. The results are
        returned as a dictionary containing the image name, embedding vector, and caption.

        Args:
            img_path (Path): Path object pointing to the image file
            img_id (int): Unique identifier for the image

        Returns:
            dict: Dictionary containing:
                - name (str): Name of the image file
                - vector (numpy.ndarray): CLIP embedding vector
                - caption (str): Generated caption for the image
                
        Raises:
            Exception: If there is an error processing the image, generating caption,
                      or creating embeddings
        """
        try:
            self.logger.debug(f"Processing image {img_path}")
            
            # Open image and convert to base64
            img = Image.open(img_path).convert(config.IMAGE_FORMAT)
            image_base64 = self.image_to_base64(img_path)

            # Construct a BAML Image Object
            image_object = {
                "base64":image_base64,
                "media_type": "image/png"
            }

            # Extract caption from image using BAML
            response = baml.b.ExtractImageCaption(image_object)
            caption = response.caption
            
            # Get CLIP embedding using image and caption
            clip_embedding = self.embeddings.get_clip_embedding(str(img_path), caption)
            
            return {
                "name": img_path.name,
                "vector": clip_embedding,
                "caption": caption
            }
                
        except Exception as e:
            self.logger.error(f"Error processing {img_path}: {e}", exc_info=True)
            return None
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


    def process_images(self, img_dir: str, num_samples: int):
        """
        Process images, extract embeddings, and prepare TensorBoard files.

        This method processes a batch of images by extracting CLIP embeddings and captions,
        and prepares files needed for TensorBoard visualization. It samples images from the
        provided directory, processes them using CLIP and BAML for captioning, and generates
        sprite images and metadata files for TensorBoard.

        Args:
            img_dir (str): Directory containing the image files to process
            num_samples (int): Number of random images to sample and process

        Raises:
            ValueError: If no images are successfully processed or sprite image creation fails
            Exception: If there are errors processing individual images
        """
        try:
            # Define paths for tensorboard
            tensorboard_dir = config.TENSORBOARD_LOG_DIR
            vectors_path = os.path.join(tensorboard_dir, 'vectors.tsv')
            metadata_path = os.path.join(tensorboard_dir, 'metadata.tsv')

            # Create tensorboard directory
            os.makedirs(tensorboard_dir, exist_ok=True)
            self.logger.info(f"TensorBoard directory ready at {tensorboard_dir}")

            # Load images and annotations
            image_files = list(Path(img_dir).glob('*.jpg'))
            sampled_files = random.sample(image_files, min(num_samples, len(image_files)))

            # Process images and collect successful entries
            successful_entries = []
            
            with tqdm(total=len(sampled_files), desc="Processing images") as pbar:
                for img_path in sampled_files:
                    try:
                        # Change to self.process_single_image
                        result = self.process_single_image(
                            img_path, 
                            int(img_path.stem.split('_')[-1])
                        )
                        
                        if result:
                            successful_entries.append({
                                'path': str(img_path),
                                'vector': result['vector'],
                                'caption': result['caption']
                            })
                                
                    except Exception as e:
                        self.logger.error(f"Error processing {img_path}: {e}")
                    finally:
                        pbar.update(1)

            if not successful_entries:
                raise ValueError("No images were successfully processed")

            # Create sprite image
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
                final_vectors.append(entry['vector'])
                metadata_row = [
                    os.path.basename(entry['path']),  # Image filename
                    entry['caption'],  # Caption
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
                writer.writerow(['Image', 'Caption'])
                for row in final_metadata:
                    cleaned_caption = ' '.join(row[1].replace('\n', ' ').split())
                    writer.writerow([row[0], cleaned_caption])

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
            self.logger.error(f"Error in process_images: {e}")
            raise


    def create_sprite_image(self, images_paths: List[str], sprite_size: int = config.SPRITE_SIZE) -> Tuple[Image.Image, List[str]]:
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