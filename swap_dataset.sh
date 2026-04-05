#!/usr/bin/env bash
# Swap the dataset in build_dataset.py from bayesft-similar to bayesft-different
set -euo pipefail

sed -i 's|desh2806/bayesft-similar|desh2806/bayesft-different|g' build_dataset.py

echo "Dataset swapped to desh2806/bayesft-different"
grep -n 'bayesft-different' build_dataset.py
