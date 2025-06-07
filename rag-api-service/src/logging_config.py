import logging
import os
import sys

# Configure log level from environment variable or default to INFO
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

def setup_logger(name):
    """
    Configure and return a logger with the specified name
    
    Args:
        name (str): Name of the logger, typically __name__
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Map string log level to logging constants
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    
    log_level = level_map.get(LOG_LEVEL, logging.INFO)
    
    # Create logger
    logger = logging.getLogger(name)
    
    # Only configure handlers if they haven't been set up yet
    if not logger.handlers:
        logger.setLevel(log_level)
        
        # Create console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(log_level)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(handler)
    
    return logger 