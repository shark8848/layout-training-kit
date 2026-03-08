# Migration Plan from layout_training_module

## Scope

Migrate capabilities from:

- `src/layout_training_module/ui.py`
- `src/layout_training_module/services/`
- `src/layout_training_module/utils/`
- `src/layout_training_module/pipeline/`
- `src/layout_training_module/skills/`

## Principles

1. Keep public API signatures stable where possible.
2. Migrate in thin slices: utility -> service -> workflow -> UI.
3. Add regression tests per migrated slice.
4. Avoid introducing runtime dependencies not required by migrated code.

## Suggested Phases

### Phase 1: Foundation

- Copy package skeleton into `src/layout_training_kit/`
- Port config loading and shared constants
- Add smoke tests for package import and basic workflow init

### Phase 2: Core pipeline

- Port reusable utility functions
- Port orchestrator/service layer
- Port pipeline entry and align I/O contracts

### Phase 3: Skills & integration

- Port skills and adapters
- Connect UI/API if needed
- Backfill docs and examples

### Phase 4: Hardening

- Add CI quality gates
- Add changelog and release notes
- Tag `v0.1.0`
