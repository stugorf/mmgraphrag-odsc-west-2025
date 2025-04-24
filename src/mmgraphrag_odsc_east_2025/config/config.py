"""Configuration settings for mmGraphRAG."""

# Standard library imports
import os
from pathlib import Path

class Config():
    """Application configuration for mmGraphRAG.

    This class contains all configuration settings for the mmGraphRAG application,
    including logging levels, data processing parameters, model settings, and file paths.

    Attributes:
        LOG_LEVEL (str): Logging level for the application, defaults to "INFO"
        SAMPLE_SIZE (int): Number of samples to process during data ingestion
        BAML_LOG (str): BAML logging configuration, defaults to "off"
        CLIP_DIM (int): Dimension of CLIP embeddings (512)
        CLIP_MODEL (str): Pre-trained CLIP model to use
        PROJECT_ROOT (str): Absolute path to project root directory
        IMAGES_DIR (str): Directory containing image files
        TENSORBOARD_LOG_DIR (str): Directory for TensorBoard logs and checkpoints
        SPRITE_SIZE (int): Size in pixels for sprite images
        SPRITE_BG_COLOR (str): Background color for sprite images
        IMAGE_FORMAT (str): Format for image processing (RGB)
    """

    # Logging
    LOG_LEVEL: str = "INFO"

    # Data Ingestion
    SAMPLE_SIZE: int = 300

    # BAML settings
    BAML_LOG: str = "off"
    
    # CLIP settings
    CLIP_DIM: int = 512
    CLIP_MODEL: str = "openai/clip-vit-base-patch32"
    # Project root and base paths
    PROJECT_ROOT: str = str(Path(__file__).parent.parent)
    IMAGES_DIR: str = str(Path(PROJECT_ROOT) / "images")

    # Tensorboard
    TENSORBOARD_LOG_DIR: str = str(Path(PROJECT_ROOT) / "tensorboard")

    # Image Processing
    SPRITE_SIZE: int = 100
    SPRITE_BG_COLOR: str = 'white'
    IMAGE_FORMAT: str = 'RGB'

# Create global config instance
try:
    config = Config()
except Exception as e:
    raise RuntimeError(f"Failed to validate configuration: {e}") 