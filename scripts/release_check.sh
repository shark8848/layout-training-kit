#!/usr/bin/env bash
set -euo pipefail

python -m pip install -U pip
pip install -e .[dev]
ruff check src tests
pytest -q

echo "release_check passed"
