import cpp_module
from relab.relab import initialize as initialize
from relab.relab import device as device
from relab.relab import config as config
import logging

# ReLab version.
version = "1.0.0-b"

# Initialize the root logger.
logging.basicConfig(level=logging.INFO)

# Build the ReLab C++ library.
cpp_module.build_cpp_library_and_wrapper()
