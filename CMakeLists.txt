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
project(relab VERSION 1.0)

# The name of the virtual environment.
SET(VENV_DIR ".venv")

# Add include directories required by shared libraries.
execute_process(COMMAND python3.12 -m pybind11 --includes OUTPUT_VARIABLE ADDITIONAL_INCLUDES)
string(REPLACE "-I" "" ADDITIONAL_INCLUDES ${ADDITIONAL_INCLUDES})
string(REPLACE " " ";" ADDITIONAL_INCLUDES ${ADDITIONAL_INCLUDES})
include_directories(${PROJECT_SOURCE_DIR}/relab/cpp/inc/ ${ADDITIONAL_INCLUDES})
include_directories(${PROJECT_SOURCE_DIR}/tests/inc/)

# Add link directories required by shared libraries.
link_directories(${PROJECT_SOURCE_DIR}/scripts/ ${PROJECT_SOURCE_DIR}/build/)

# Add the torch library.
list(APPEND CMAKE_PREFIX_PATH "${PROJECT_SOURCE_DIR}/${VENV_DIR}/lib/python3.12/site-packages/torch")
link_directories("${PROJECT_SOURCE_DIR}/${VENV_DIR}/lib")
set(ENV{MKLROOT} "${PROJECT_SOURCE_DIR}/${VENV_DIR}/lib")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")

# Add the zlib library.
find_package(ZLIB REQUIRED)

# Add python library.
find_package(Python 3.12 EXACT REQUIRED COMPONENTS Interpreter Development)

# Add python extension package.
find_package(PythonExtensions REQUIRED)

# Create a list containing all libraries shared by all executables and shared libraries of the project.
list(APPEND ALL_LIBRARIES
    ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY}
    ZLIB::ZLIB
    ${Python_LIBRARIES}
    pthread
    stdc++fs
)

# Download and make Google test available.
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
enable_testing()

# Setup CMake install path.
include(GNUInstallDirs)
if (APPLE)
  set(rbase "@loader_path")
else ()
  set(rbase "$ORIGIN")
endif ()
file(RELATIVE_PATH lib_dir
     "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}"
     "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")
set(CMAKE_INSTALL_RPATH "${rbase};${rbase}/${libdir}" CACHE STRING "Install RPATH")

# Add the shared library: librelab.so
add_library(relab SHARED
    relab/cpp/src/agents/memory/frame_storage.cpp
    relab/cpp/src/agents/memory/replay_buffer.cpp
    relab/cpp/src/agents/memory/priority_tree.cpp
    relab/cpp/src/agents/memory/frame_buffer.cpp
    relab/cpp/src/agents/memory/compressors.cpp
    relab/cpp/src/agents/memory/data_buffer.cpp
    relab/cpp/src/agents/memory/experience.cpp
    relab/cpp/src/helpers/thread_pool.cpp
    relab/cpp/src/helpers/serialize.cpp
    relab/cpp/src/helpers/debug.cpp
    relab/cpp/src/helpers/deque.cpp
    relab/cpp/src/helpers/timer.cpp
    relab/cpp/src/helpers/torch.cpp
)
target_link_libraries(relab PRIVATE ${ALL_LIBRARIES})
install(TARGETS relab LIBRARY DESTINATION relab)

# Add the wrapper shared library: librelab_wrapper.so
add_library(cpp MODULE relab/cpp/pybind11_wrapper.cpp)
target_link_libraries(cpp relab ${ALL_LIBRARIES})
python_extension_module(cpp)
install(TARGETS cpp LIBRARY DESTINATION relab)

# Create an executable running unit tests.
add_executable(all_tests
    tests/src/agents/memory/test_replay_buffer.cpp
    tests/src/agents/memory/test_priority_tree.cpp
    tests/src/agents/memory/test_frame_buffer.cpp
    tests/src/agents/memory/test_data_buffer.cpp
    tests/src/helpers/test_deque.cpp
    tests/src/relab_test.cpp
)
target_link_libraries(all_tests relab ${ALL_LIBRARIES} GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(all_tests)

# Add the testing executable with access to the project's shared libraries.
add_executable(main_test tests/main.cpp)
target_link_libraries(main_test PRIVATE ${ALL_LIBRARIES} relab)
