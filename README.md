# Neural Collaborative Filtering Recommender System

This repository contains the implementation of a **Neural Collaborative Filtering (NCF)** model designed to facilitate user exchanges on platforms like Ratebeer.com. The project builds upon prior research and incorporates advanced techniques to improve recommendation accuracy for niche exchange platforms.

## Project Overview

The model leverages the strengths of deep learning to address the limitations of traditional recommendation approaches like matrix factorization. It optimizes recommendations for implicit feedback scenarios using Bayesian Personalized Ranking (BPR) as the loss function.

## Key Features

- **NCF Model Architecture**: Implements a neural network-based approach for non-linear and higher-order relationships between users and items.
- **Bayesian Personalized Ranking (BPR)**: Optimizes ranking performance rather than prediction error.
- **Hyperparameter Tuning**: Includes search over embedding dimensions, batch sizes, learning rates, and network configurations.
- **Social and Temporal Dynamics**: Supports additional embeddings for social bias and time-based features.

# Project Structure

```shell
├── evaluate.py            # Evaluation script for computing AUC and ranking metrics
├── load_data.py          # Prepares and loads the dataset for training and evaluation
├── loss.py               # Implements the BPR loss function
├── main.py              # Main script for running experiments and hyperparameter tuning
├── ncf.py               # Defines the Neural Collaborative Filtering (NCF) model
└── train_ncf.py         # Training script for the NCF model


