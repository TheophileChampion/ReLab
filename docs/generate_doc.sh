#!/bin/bash
# This script must be run from the project root.

./docs/generate_readme.sh
doxygen ./Doxyfile
