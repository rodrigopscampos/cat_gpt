#!/bin/bash
set -euo pipefail

echo "Running Ruff checks..."
poetry run ruff check . --fix

echo "Running Pyright..."
poetry run pyright .

echo "All good!"