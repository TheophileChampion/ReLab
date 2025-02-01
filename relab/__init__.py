import logging

from relab.relab import config as config
from relab.relab import device as device
from relab.relab import initialize as initialize

# ReLab version.
version = "1.0.0-b"

# Initialize the root logger.
logging.basicConfig(level=logging.INFO)
