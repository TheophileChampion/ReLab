from relab.relab import initialize as initialize
from relab.relab import device as device
from relab.relab import config as config
from relab.relab import build_cpp_library_and_wrapper as build_cpp_library_and_wrapper
from os.path import join
import logging
import os

# ReLab version.
version = "1.0.0-b"

# Initialize the root logger.
logging.basicConfig(level=logging.INFO)

# Set the environment variable:
# "ROOT_DIRECTORY", "BUILD_DIRECTORY", and "CPP_MODULE_DIRECTORY".
root_directory = join(os.getcwd())
os.environ["ROOT_DIRECTORY"] = root_directory
os.environ["BUILD_DIRECTORY"] = join(root_directory, "build")
os.environ["CPP_MODULE_DIRECTORY"] = join(root_directory, "relab")

# Build the C++ library and the python module wrapping the library.
build_cpp_library_and_wrapper()
