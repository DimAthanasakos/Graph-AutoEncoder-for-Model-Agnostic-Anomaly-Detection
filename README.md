# Graph Autoencoder for Model Agnostic Anomaly Detection

[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A PyTorch implementation of Graph Autoencoders for model-agnostic anomaly detection in particle collision data at the LHC. This repository implements the Graph Autoencoder from the paper "Graph theory inspired anomaly detection at the LHC" along with several other architectures for unsupervised and weakly-supervised anomaly detection.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Training Methods](#training-methods)
- [Pipeline](#pipeline)
- [Citation](#citation)

## Overview

This project provides tools for detecting anomalous events in high-energy physics data using graph neural networks. The code can work with both hadron-level and subjet-level particle data.

The implementation is optimized for the LHC Olympics R&D dataset and includes preprocessing tools for jet clustering and feature extraction.

## Features

- **Multiple Model Architectures**:
  - RelGAE (Relational Graph Autoencoder)
  - EdgeNet
  - Standard Autoencoder (AE)
  - Variational Autoencoder (VAE)

- **Flexible Input Types**:
  - Hadron-level particles
  - Subjets (requires preprocessing)

- **Training Modes**:
  - Unsupervised anomaly detection
  - Weakly supervised using sideband regions

- **Graph Types**:
  - Fully connected graphs
  - Laman graphs
  - Unique-k graphs

## Installation

1. **Install dependencies**:
   ```bash
   # For standard ML analysis
   source ./init_perlmutter.sh
   
   # For preprocessing (subjet clustering)
   source ./init_perlmutter_heppy.sh
   ```

2. **Download the LHC Olympics R&D dataset**:
   - Download from [Zenodo](https://zenodo.org/records/6466204)
   - Place in appropriate directory structure

## Quick Start

1. **Configure your experiment**:
   ```bash
   # Edit config/config.yaml with your parameters
   nano config/config.yaml
   ```

2. **Run the analysis**:
   ```bash
   python -u analysis/steer_analysis.py -c config/config.yaml
   ```

3. **Check results**:
   - Results are saved in `Results/` directory
   - Plots (although not all) are generated in `plots_gae/`

## Usage

### Basic Usage

```bash
# Run with default configuration
python -u analysis/steer_analysis.py -c config/config.yaml

# Run with specific model
python -u analysis/steer_analysis.py -c config/config.yaml --model RelGAE

```

### Configuration

Most parameters are controlled through `config/config.yaml`. Key parameters include:

- `n_train`, `n_val`: Number of training/validation samples
- `n_part`: Number of particles per jet
- `subjets`: Use subjets (1) or hadrons (0)
- `models`: List of models to train
- `unsupervised`: Training mode
- `s_over_b`: Signal-to-background ratio for unsupervised training

## Project Structure

```
gae_for_anomaly/
├── analysis/                # Main analysis scripts
│   ├── steer_analysis.py    # Main entry point
│   ├── ml_analysis.py       # ML pipeline
│   ├── gae_train.py         # GAE training
│   ├── ml_anomaly.py        # Anomaly detection
│   ├── process_subjets.py   # Subjet preprocessing
│   ├── utils.py             # Utility functions
│   └── models/              # Model implementations
├── config/                  # Configuration files
│   └── config.yaml          # Main configuration
├── preprocessing/           # Preprocessing results
├── Results/                 # Analysis results
├── plots_gae/               # Generated plots
└── init_perlmutter*.sh      # Environment setup scripts
```

## Configuration

The `config/config.yaml` file contains all model-specific parameters:

```yaml
# Example configuration
n_train: 800
n_val: 100
n_part: [10]
subjets: 1
models: ['RelGAE']

RelGAE:
  graph_types: ['laman']
  batch_size: 1
  epochs: 2
  learning_rate: 0.003
  unsupervised: True
  s_over_b: 0.03
```

## Training Methods

### Unsupervised Training
- Trains on the full phase-space
- Uses signal-to-background ratio (`s_over_b`) 
- Traditional anomaly search with an autoencoder based model

### Weakly Supervised Training
- Uses sideband regions for training
- Tests on signal region (SR)
- Although we can in principle use a `s_over_b` ratio for WS search, currently this is not supported.

## Pipeline

The analysis follows this pipeline:

1. **Data Loading** (`steer_analysis.py`)
   - Load and preprocess particle data
   - Apply cuts and selections

2. **ML Analysis** (`ml_analysis.py`)
   - Feature extraction and normalization
   - Train/test splitting

3. **Model Training** (`gae_train.py`)
   - Train selected models
   - Save model checkpoints

4. **Anomaly Detection** (`ml_anomaly.py`)
   - Generate anomaly scores
   - Evaluate performance metrics

5. **Visualization** (`plot_script/`)
   - Generate performance plots
   - Create summary statistics


## Citation

If you use this code in your research, please cite:

```bibtex
@article{Araz:2025oax,
    author = "Araz, Jack Y. and Athanasakos, Dimitrios and Ploskon, Mateusz and Ringer, Felix",
    title = "{Graph theory inspired anomaly detection at the LHC}",
    eprint = "2506.19920",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    reportNumber = "INT-PUB-25-019, YITP-SB-2025-11",
    month = "6",
    year = "2025"
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
