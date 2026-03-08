# Release Guide

## Pre-release

```bash
cd /home/layout-training-kit
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[dev]
./scripts/release_check.sh
```

## GitHub initial publish

```bash
git init
git add .
git commit -m "chore: init layout-training-kit scaffold"
git branch -M main
git remote add origin git@github.com:<your-org>/layout-training-kit.git
git push -u origin main
```

## First tag

```bash
git tag v0.1.0
git push origin v0.1.0
```
