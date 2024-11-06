import logging
import os

def setup_logging(log_file):    
    """Set up logging.""" 
    print(f"Logging at file: {log_file}")
    logging.basicConfig(filename=log_file, 
                        level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')

def clear_logging(log_file):
    """Clear the logging file."""
    with open(log_file, 'w'):
        pass

def reset_logging():
    """Clear all logging handlers to release the log file."""
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)