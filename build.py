"""
Module installing python dependencies and building ReLab's C++ extension.
"""

import invoke

if __name__ == "__main__":

    # Build C++ extensions.
    invoke.run(
        "source .venv/bin/activate "
        "&& pip install scikit-build cmake ninja pybind11==2.13.6 "
        "&& python build_extension.py"
    )
