from relab.relab import initialize, device, config
from os.path import join, dirname
import logging
import os

# ReLab version.
version="1.0.0-b"

# Initialize the root logger.
logging.basicConfig(level=logging.INFO)

# Set the environment variable: "ROOT_DIRECTORY", "BUILD_DIRECTORY", and "CPP_MODULE_DIRECTORY".
os.environ["ROOT_DIRECTORY"] = join(dirname(__file__), "..")
os.environ["BUILD_DIRECTORY"] = join(os.environ["ROOT_DIRECTORY"], "build")
os.environ["CPP_MODULE_DIRECTORY"] = join(os.environ["ROOT_DIRECTORY"], "relab")

# Build the C++ library and the python module wrapping the library.
relab.build_cpp_library_and_wrapper()
