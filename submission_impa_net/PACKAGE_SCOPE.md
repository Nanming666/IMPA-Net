# Package Scope

This folder is a **sample submission package**, not the full research codebase.

## Intentionally Preserved

- Meteorology-aware multi-channel input construction
- Radar-dominant encoding with auxiliary meteorological fusion
- Multi-scale spatial feature extraction
- Temporal sequence mixing
- Dynamic extreme-aware loss formulation
- NetCDF-based example data loading

## Intentionally Omitted

- Full training framework
- Distributed training
- Experiment manager and hooks
- Hyperparameter configuration system
- Full evaluation suite
- Logging and checkpoint management
- Data preprocessing scripts for the complete dataset
- Internal engineering utilities unrelated to the paper's core method

## Rationale

The goal of this package is to share:

1. the central method idea
2. the expected example data structure
3. a clean, compact script that is easy to inspect

This keeps the public-facing material concise while still reflecting the method contribution described in the manuscript.
