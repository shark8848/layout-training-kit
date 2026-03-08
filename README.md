# layout-training-kit

Open-source toolkit for document layout training workflows.

## Features

- Modular service-oriented workflow design
- Reusable utilities for data preparation and training pipeline orchestration
- CI-ready Python project scaffold
- Clear migration path from `layout_training_module`

## Quick Start

```bash
cd /home/layout-training-kit
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[dev]
pytest
```

## Project Structure

```text
layout-training-kit/
├── src/layout_training_kit/
├── tests/
├── docs/
├── scripts/
└── .github/workflows/
```

## Roadmap

- [ ] Migrate core capabilities from `layout_training_module`
- [ ] Add training dataset adapters
- [ ] Add model evaluation benchmark commands
- [ ] Publish first `v0.1.0`
