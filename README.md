# Cyber-Nano-GPT

A compact GPT-style Transformer project for modeling structured network-security telemetry.

The project converts CIC-IDS2018 network-flow records into temporally consistent token sequences using PySpark and trains a custom Transformer implemented from scratch in PyTorch. The current prototype includes preprocessing, sequence construction, model components, and an end-to-end training pipeline.

**Implemented:** PySpark preprocessing · 19-feature telemetry representation · boundary-aware sequence construction · custom causal self-attention and Transformer blocks · next-token training pipeline

A preliminary 100-step training run reduced training loss from 10.5 to 0.5, providing an initial sanity check of the end-to-end implementation.

## Project Goal

The goal is to evaluate whether next-token prediction over structured security telemetry can provide useful signals for anomaly-oriented analysis.

## Modeling Idea

Instead of treating telemetry analysis as a fixed-label classification problem, this project explores next-token prediction over structured event sequences. The idea is to model what “normal continuation” looks like in telemetry and then inspect whether unusual token-level behavior can provide a useful signal for suspicious activity.

## Implemented so far

- Built a PySpark-based preprocessing pipeline for CIC-IDS2018 network-flow data.
- Reduced the raw feature space from 80 fields to 19 curated fields for modeling.
- Applied log-binning and converted each record into a fixed-order token sequence for GPT-style next-token modeling.
- Preserved temporal sequence integrity by splitting traffic into contiguous benign segments and preventing training windows from crossing attack/benign boundaries.
- Discarded fragments shorter than 13 rows that could not form a complete training window, avoiding artificially stitched sequences across unrelated traffic segments.
- This retained 81.54% of benign traffic (703,533 rows) while prioritizing sequence fidelity over raw training volume.
- Implemented core GPT-style Transformer components in PyTorch, including causal self-attention, Transformer blocks, token embeddings, and positional embeddings.
- Verified core architecture behavior through parameter-count and forward-pass checks.
- Prepared the training data interface by building a benign-only vocabulary, adding fallback handling for unseen feature values, and creating an iterable dataset loader for fixed-length next-token modeling.

## Current status

The preprocessing pipeline, training-data interface, custom Transformer components, and end-to-end training pipeline are implemented. A preliminary 100-step run was used to validate that the training pipeline behaves as expected.

The next step is systematic held-out evaluation and testing whether token-level surprise or perplexity provides a useful anomaly signal.

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

## Next steps

- Run longer baseline training and evaluate on held-out telemetry.
- Test whether token-level surprise or perplexity provides a useful anomaly signal.
- Improve experiment reporting and reproducibility.

## Possible extensions

These are later-stage ideas, not current core claims:

- Add a lightweight explanation layer for flagged sequences after the baseline modeling pipeline is stable.
- Explore retrieval-augmented analysis using external security knowledge as an optional downstream extension.
