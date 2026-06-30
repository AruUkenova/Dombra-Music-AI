# Dombra-Music-AI

Official implementation of the paper:

**Transcription of Audio Signals Using Convolutional-Recurrent Neural Networks**

## Overview

This repository contains the source code for automatic transcription of Kazakh dombra recordings into musical notation using a Convolutional Recurrent Neural Network (CRNN) and Constant-Q Transform (CQT) features.

## Repository Structure

```
.
├── src/
│   ├── model.py          # CRNN training
│   ├── datasetmake.py    # Synthetic dataset generation
│
├── example_data/         # Example audio and labels
├── requirements.txt      # Python dependencies
├── LICENSE
└── README.md
```

## Requirements

Install the required Python packages with:

```bash
pip install -r requirements.txt
```

## Dataset

The model was trained using a synthetic dataset consisting of 2,000 generated audio recordings of dombra performances.

Due to copyright restrictions on the original recordings, the complete dataset is not publicly distributed. However, the repository provides the scripts required to reproduce the synthetic dataset.

## Reproducibility

The repository includes:

- Synthetic dataset generation
- CRNN model implementation
- Training pipeline
- Example input data

## Citation

If you use this code in your research, please cite the associated publication.
