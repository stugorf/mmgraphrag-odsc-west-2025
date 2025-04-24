"""Embedding utilities for generating semantic embeddings from images and text.

This module provides functionality for generating semantic embeddings from images and text
using CLIP (Contrastive Language-Image Pre-Training) models. The embeddings can be used
for multimodal similarity search and retrieval.

The main class SemanticEmbeddings handles loading CLIP models and generating embeddings
from image-text pairs.

Example:
    embedder = SemanticEmbeddings()
    embedding = embedder.get_clip_embedding(
        image_path="path/to/image.jpg",
        caption="A description of the image"
    )

Attributes:
    CLIP_MODEL (str): Pre-trained CLIP model identifier from config
    CLIP_DIM (int): Dimension of CLIP embeddings from config
"""

# Standard library imports
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel

# Third party imports   
from PIL import Image

# Local imports
from mmgraphrag_odsc_aib2025.config import config
from mmgraphrag_odsc_aib2025.utils.logger import setup_logger

class SemanticEmbeddings():
    """Class for generating semantic embeddings from images and text using CLIP.

    This class provides methods to generate semantic embeddings from images and optional
    text captions using CLIP (Contrastive Language-Image Pre-Training) models. The embeddings
    can be used for multimodal similarity search and retrieval.

    The class uses the CLIP model and processor specified in the config to generate embeddings
    of dimension CLIP_DIM. When both image and caption are provided, it combines their 
    embeddings through averaging.

    Attributes:
        logger: Logger instance for this class

    Example:
        embedder = SemanticEmbeddings()
        embedding = embedder.get_clip_embedding(
            image_path="path/to/image.jpg",
            caption="A description of the image"
        )
    """

    def __init__(self):
        self.logger = setup_logger(__name__)

    def get_clip_embedding(self, image_path: str, caption: str = None) -> np.ndarray:
            """Get CLIP embedding for image and optional caption.
            
            Args:
                image_path: Path to image file
                caption: Optional caption text to combine with image embedding
                
            Returns:
                numpy.ndarray: Combined CLIP embedding
            """
            try:
                clip_model = CLIPModel.from_pretrained(config.CLIP_MODEL)
                clip_processor = CLIPProcessor.from_pretrained(config.CLIP_MODEL)
            except Exception as e:
                print(f"Error initializing CLIP: {e}")
                clip_model = None
                clip_processor = None
            try:
                if clip_model is None or clip_processor is None:
                    raise ValueError("CLIP model not initialized")
                
                # Load and process image
                image = Image.open(image_path).convert('RGB')
                inputs = clip_processor(images=image, return_tensors="pt")
                
                with torch.no_grad():
                    # Get image embedding
                    image_features = clip_model.get_image_features(**inputs)
                    image_embedding = image_features.cpu().numpy()[0]
                    
                    # If caption provided, get text embedding and combine
                    if caption:
                        text_inputs = clip_processor(text=caption, return_tensors="pt", padding=True)
                        text_features = clip_model.get_text_features(**text_inputs)
                        text_embedding = text_features.cpu().numpy()[0]
                        
                        # Average the embeddings
                        combined_embedding = (image_embedding + text_embedding) / 2
                        return combined_embedding
                        
                    return image_embedding
                
            except Exception as e:
                print(f"Error getting CLIP embedding: {e}")
                raise