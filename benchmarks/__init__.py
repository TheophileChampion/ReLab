from benchmarks.benchmarks import initialize, device, config
from os.path import join, abspath, dirname
import os

# Set the environment variable: "ROOT_DIRECTORY", "BUILD_DIRECTORY", and "CPP_MODULE_DIRECTORY".
os.environ["ROOT_DIRECTORY"] = join(dirname(abspath(__file__)), "..")
os.environ["BUILD_DIRECTORY"] = join(os.environ["ROOT_DIRECTORY"], "build")
os.environ["CPP_MODULE_DIRECTORY"] = join(os.environ["ROOT_DIRECTORY"], "benchmarks", "agents", "memory")

# Build the C++ library and the python module wrapping the library.
benchmarks.build_cpp_library_and_wrapper()
