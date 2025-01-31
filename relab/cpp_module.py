import os
from os.path import isfile, join

import invoke

from relab.helpers.FileSystem import FileSystem


def build_cpp_library_and_wrapper(
    cpp_library_name: str = "relab", python_module_name: str = "cpp"
) -> None:
    """!
    Build the C++ shared library and the python module wrapping the library.
    @param cpp_library_name: the name of the shared library to create
    @param python_module_name: the name of the python module
    """

    # Check if the shared libraries already exist.
    build_directory = os.environ["BUILD_DIRECTORY"]
    shared_library = join(build_directory, f"lib{cpp_library_name}.so")
    module_directory = os.environ["CPP_MODULE_DIRECTORY"]
    files = FileSystem.files_in(module_directory, rf"^{python_module_name}.*")
    if isfile(shared_library) and len(files) != 0:
        return

    # Create the shared libraries.
    librelab_wrapper = "./build/librelab_wrapper.so"
    extension = "python3.12-config --extension-suffix"
    invoke.run(
        f"mkdir -p {build_directory} && cd {build_directory} && cmake .. && make && cd .. "
        f"&& mv {librelab_wrapper} {module_directory}/{python_module_name}`{extension}`"
    )


# Set the environment variable:
# "ROOT_DIRECTORY", "BUILD_DIRECTORY", and "CPP_MODULE_DIRECTORY".
root_directory = join(os.getcwd())
os.environ["ROOT_DIRECTORY"] = root_directory
os.environ["BUILD_DIRECTORY"] = join(root_directory, "build")
os.environ["CPP_MODULE_DIRECTORY"] = join(root_directory, "relab")

# Build the C++ library and the python module wrapping the library.
build_cpp_library_and_wrapper()
