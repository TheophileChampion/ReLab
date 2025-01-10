# Specify the C++ and CUDA compilers.
set(CMAKE_CXX_COMPILER "/usr/bin/g++")
set(CMAKE_CUDA_ARCHITECTURES "native")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.6/bin/nvcc")

# Specify the version of C++ to use.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add compiler options.
add_compile_options(-O3 -Wall -Werror -Wl,-rpath,.)

# Specify the CMake version, as well as the project name and version.
cmake_minimum_required(VERSION 3.10)
project(benchmarks VERSION 1.0)

# The name of the virtual environment.
SET(VENV_DIR "venv-benchmarks")

# Add the zlib library.
find_package(ZLIB REQUIRED)

# Add the torch library.
list(APPEND CMAKE_PREFIX_PATH "${PROJECT_SOURCE_DIR}/${VENV_DIR}/lib/python3.12/site-packages/torch")
link_directories("${PROJECT_SOURCE_DIR}/${VENV_DIR}/lib")
set(ENV{MKLROOT} "${PROJECT_SOURCE_DIR}/${VENV_DIR}/lib")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")

# Add include directories required by shared libraries.
execute_process(COMMAND python3.12 -m pybind11 --includes OUTPUT_VARIABLE ADDITIONAL_INCLUDES)
string(REPLACE "-I" "" ADDITIONAL_INCLUDES ${ADDITIONAL_INCLUDES})
string(REPLACE " " ";" ADDITIONAL_INCLUDES ${ADDITIONAL_INCLUDES})
include_directories(${PROJECT_SOURCE_DIR}/benchmarks_cpp/inc/ ${ADDITIONAL_INCLUDES})

# Add link directories required by shared libraries.
link_directories(${PROJECT_SOURCE_DIR}/scripts/ ${PROJECT_SOURCE_DIR}/scripts/build/)

# Add the shared library: libbenchmarks.so
add_library(benchmarks SHARED
    benchmarks_cpp/src/compressor.cpp
    benchmarks_cpp/src/frame_storage.cpp
    benchmarks_cpp/src/replay_buffer.cpp
    benchmarks_cpp/src/frame_buffer.cpp
    benchmarks_cpp/src/data_buffer.cpp
    benchmarks_cpp/src/priority_tree.cpp
    benchmarks_cpp/src/experience.cpp
    benchmarks_cpp/src/deque.cpp
)
target_link_libraries(benchmarks PRIVATE
    ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY}
    pthread
)

# Add the wrapper shared library: libbenchmarks_wrapper.so
add_library(benchmarks_wrapper SHARED benchmarks_cpp/pybind11_wrapper.cpp)
target_link_libraries(benchmarks_wrapper PRIVATE
    benchmarks
    ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY}
    pthread
)

# Add the testing executable with access to the shared library: libbenchmarks.so
add_executable(test benchmarks_cpp/test/main.cpp)
target_link_libraries(test PRIVATE
    benchmarks_wrapper benchmarks
    ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY}
    ZLIB::ZLIB
    pthread
    python3.12
)

# TODO

# Add the nvcomp library.
# list(APPEND CMAKE_PREFIX_PATH "${PROJECT_SOURCE_DIR}/${VENV_DIR}/lib/python3.12/site-packages/nvidia")
# find_package(nvcomp REQUIRED)

# old include_directories(high_level_quickstart_example PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

# add_compile_definitions("THRUST_CUB_WRAPPED_NAMESPACE=nvcomp")

# TODO include_directories(high_level_quickstart_example PRIVATE ${PROJECT_SOURCE_DIR}/${VENV_DIR}/lib/python3.12/site-packages/nvidia/nvcomp/include/)
# TODO link_directories(${PROJECT_SOURCE_DIR}/${VENV_DIR}/lib/python3.12/site-packages/nvidia/nvcomp/)
# TODO add_executable(high_level_quickstart_example benchmarks_cpp/test/high_level_quickstart_example.cpp)
# TODO target_link_libraries(high_level_quickstart_example PRIVATE nvcomp CUDA::cudart)
