import sys
import logging
from mmgraphrag_odsc_aib2025.config import config

def setup_logger(name: str) -> logging.Logger:
    """Configure and return a logger instance.

    This function sets up a logger with consistent formatting and configuration across
    the application. It creates a logger instance with the specified name and configures
    it based on the LOG_LEVEL setting from the application config.

    Args:
        name (str): Name for the logger instance, typically __name__ of calling module

    Returns:
        logging.Logger: Configured logger instance with console handler and formatter

    Example:
        logger = setup_logger(__name__)
        logger.info("Application started")
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        # Convert string log level to logging constant
        log_level = getattr(logging, config.LOG_LEVEL.upper())
        logger.setLevel(log_level)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    return logger