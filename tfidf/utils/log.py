import logging
import os

if not os.path.exists('./log'):
    os.makedirs('./log')
    
def setup_logger(name_):
    """Set up and configure logger for debugging"""
    # Configure logger
    logger = logging.getLogger(name_)
    logger.setLevel(logging.DEBUG)

    # Create handlers
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(f'./log/{name_}.log')
    
    # Set levels
    console_handler.setLevel(logging.INFO)
    file_handler.setLevel(logging.DEBUG)

    # Create formatters and add it to handlers
    log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(log_format)
    file_handler.setFormatter(log_format)

    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

# Create global logger instance
logger = setup_logger('default')
