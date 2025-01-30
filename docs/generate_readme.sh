#!/bin/bash
# This script must be run from the project root.

echo "# 🚀 Getting Started" > ./docs/README.md

< ./README.md tr "\n" "\r" | sed -e "s/.*-------\r\(.*\)<!-- toc -->.*/\1/" | tr "\r" "\n" >> ./docs/README.md

< ./README.md sed -n -e '/# Installation/,$p' \
            | sed -e "s/\[installed\](\#installation)/installed/" \
            | tr "\n" "\r" | sed -r -e "s/# Documentation([^#]|\r)*//" | tr "\r" "\n" >> ./docs/README.md
