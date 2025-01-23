#!/bin/bash

echo "## 🚀 ReLab" > ./README.md
cat ../README.md | sed -n -e '/## Installation/,$p' \
                 | sed -e "s/\[LICENSE\](LICENSE)/LICENSE/" \
                 | tr "\n" "\r" | sed -r -e "s/## Documentation([^#]|\r)*//" | tr "\r" "\n" >> ./README.md
