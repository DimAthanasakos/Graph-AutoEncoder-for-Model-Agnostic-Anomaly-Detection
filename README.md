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


1. **Download the LHC Olympics R&D dataset**:
   - Download from [Zenodo](https://zenodo.org/records/6466204)
   - Place in appropriate directory structure. If using the perlmutter cluster, store in your personal pscratch dir.

## Quick Start

1. **Change all the mentions of /pscratch/sd/d/dimathan/ to your own dir**

2. **Request a gpu node, replace alice_g with your own allocation at perlmutter**
   ```bash
   salloc --nodes 1 --qos interactive --time 04:00:00 --constraint gpu --gpus 4 --account=alice_g
   ```

2. **Install dependencies**:
   ```bash
   # For standard ML analysis
   source ./init_perlmutter.sh
   
   # For preprocessing (subjet clustering)
   source ./init_perlmutter_heppy.sh
   ```

3.  **Configure your experiment**:
   ```bash
   # Edit config/config.yaml with your parameters
   nano config/config.yaml
   ```

4. **Run the analysis**:
   ```bash
   python analysis/steer_analysis.py -c config/config.yaml
   ```

5. **Run asynchronously for longer**:
   # Edit the script in sbatch_commands/run1.sh 
   ```bash
   sbatch sbatch_commands/run1.sh
   ```

6. **Check results**:
   - Output to the terminal from sbatch runs are saved in `Results/` directory
   - (Some) Plots are generated in `Plots`

## Usage

### Basic Usage

```bash
# Run with default configuration
python analysis/steer_analysis.py -c config/config.yaml

# Take a good look at the flags defined at the end of steer_analysis.py.

# Use a specific number of runs and then print the summary statistics via the flag --n_runs 
python analysis/steer_analysis.py -c config/config.yaml  --n_runs 3 

# By default, we use torch.compile(model) which creates an overhead of ~2 minutes for the first epoch. 
# For small-scale testing, e.g. 10k events or a small number of epochs, overwrite this by the flag -ncom 
python analysis/steer_analysis.py -c config/config.yaml -ncom
```

### Configuration

Most parameters are controlled through `config/config.yaml`, some via flags passed to ` steer_analysis.py.`
Key parameters include:

- `n_train`, `n_val`: Number of training/validation samples
- `n_part`: Number of particles/subjets per jet
- `subjets`: Use subjets (1) or hadrons (0)
- `dataset`: Load the qq (standard RnD) or the qqq LHCO dataset
- `models`: List of models to train
- `unsupervised`: Training mode
- `s_over_b`: Signal-to-background ratio for unsupervised training
- `-ncom`: Flag, whether to compile the ml model. By default yes. 

## Project Structure

```
Graph_AutoEncoder_for_Model_Agnostic_Anomaly_Detection/
├── analysis/                    # Main analysis scripts
│   ├── steer_analysis.py        # Main entry point
│   ├── ml_analysis.py           # ML pipeline
│   ├── gae_train.py             # GAE training
│   ├── ml_anomaly.py            # Anomaly detection
│   ├── utils.py                 # Utility functions
│   ├── data_preprocessing.ipynb # Basic data analysis. Preprocess the raw LHCO event data, cluster into 2 jets, save it in the pscratch dir.
│   ├── process_subjets.py       # Custom subjet preprocessing using the heppy library
│   └── models/models.py         # Model implementations
├── config/                   
│   └── config.yaml              # Main configuration
├── sbatch_commands/             # sbatch script to run code asynchronously
├── Results/                     # Store the terminal output when using sbatch
├── Plots/                       # Generated plots
├── init_perlmutter.sh           # Environment setup script for ML training
└── init_perlmutter_heppy.sh     # Environment setup script for subjet preprocessing 
```

## Configuration

The `config/config.yaml` file contains all model-specific parameters:

```yaml
# Example configuration
n_train: 800
n_val: 100
n_part: [10]
subjets: 1
dataset_type: 'qq'
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

### Weakly Supervised Training: Not recommended, only for testing purposes
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

4. **ML Architecture** (`models/models.py`)
   - Modify the ML models used

5. **Anomaly Detection** (`ml_anomaly.py`)
   - Generate anomaly scores
   - Evaluate performance metrics

The first time a particular combination of `subjets`, `n_part`, `graph_types` is passed, `steer_analysis.py` will
load the raw 2-jet particle-level data (non-graphed) and create the graphs and store them in the pscratch directory. For subsequent runs
it will load the graphs.

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
