# Cyber-Nano-GPT

A small GPT-style Transformer project for security telemetry modeling.

## What it is

Cyber-Nano-GPT is an ongoing project for building a compact GPT-style Transformer over structured security telemetry. The current focus is on converting network-flow records into token sequences, implementing core Transformer components in PyTorch, and using the resulting pipeline for controlled next-token modeling experiments.

## Project Goal

The near-term goal is to build a working next-token modeling baseline for security telemetry and evaluate whether token-level prediction signals can be useful for anomaly-oriented analysis.

## Modeling Idea

Instead of treating telemetry analysis as a fixed-label classification problem, this project explores next-token prediction over structured event sequences. The idea is to model what “normal continuation” looks like in telemetry and then inspect whether unusual token-level behavior can provide a useful signal for suspicious activity.

## Implemented so far

- Built a PySpark-based preprocessing pipeline for CIC-IDS2018 network-flow data.
- Reduced the raw feature space from 80 fields to 19 curated fields for modeling.
- Applied log-binning and converted each record into a fixed-order token sequence for GPT-style next-token modeling.
- Enforced strict temporal sequence integrity via state-based sequence segmentation (Hard Boundary Reset):
  - While the raw dataset naturally contains massive high-integrity "Golden Windows" (up to 280k contiguous rows), significant portions are interleaved with attack sessions.
  - To prevent the model from learning fake or "stitched" causal relationships, strictly isolated contiguous benign blocks and discarded any sub-batch fragments (< 13 rows) that could not form a full training window.
  - This engineering trade-off prioritized sequence fidelity over raw volume, successfully preserving 81.54% of total benign traffic (703,533 rows) while guaranteeing 100% causal purity.
- Implemented core GPT-style Transformer components in PyTorch, including causal self-attention, Transformer blocks, token embeddings, and positional embeddings.
- Verified core architecture behavior through parameter-count and forward-pass checks.
- Prepared the training data interface by building a benign-only vocabulary, adding fallback handling for unseen feature values, and creating an iterable dataset loader for fixed-length next-token modeling.


## Current technical direction

The current repository centers on three pieces:

1. **Telemetry preprocessing**
   - Transform raw network-flow records into compact, structured token sequences.

2. **Custom Transformer implementation**
   - Build and verify GPT-style components directly in PyTorch for controlled experiments.

3. **Baseline next-token modeling**
   - Train and inspect sequence models before adding more ambitious downstream layers.

## Current status

The preprocessing pipeline, core model components, and training-data interface are implemented. The project is currently focused on running and refining end-to-end baseline training and evaluation.

## Repository structure

```text
cyber-nano-gpt/
├── .dvc/          # data versioning configuration
├── config/        # experiment and preprocessing configuration
├── data/raw/      # raw input data references
├── doc/           # notes and supporting documentation
├── notebooks/     # exploratory analysis and prototyping
├── scripts/       # preprocessing and utility scripts
├── src/           # core source code
├── README.md
└── LICENSE
```

## Planned next

- Run a clean end-to-end baseline training pipeline.
- Evaluate next-token prediction behavior on held-out telemetry data.
- Test whether token-level surprise or perplexity can serve as a useful anomaly signal.
- Improve result reporting and experiment documentation.

## Possible extensions

These are later-stage ideas, not current core claims:

- Add a lightweight explanation layer for flagged sequences after the baseline modeling pipeline is stable.
- Explore retrieval-augmented analysis using external security knowledge as an optional downstream extension.

## Notes

This repository is intended to reflect the current implementation state of the project. Planned extensions are listed separately from the components already built.