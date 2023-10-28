# AG News Text Classification with LSTM

[![Skill Icons](https://skillicons.dev/icons?i=py,pytorch&perline=2)](https://skillicons.dev)

This repository provides a Jupyter notebook that explores text classification on the AG News dataset using LSTM (Long Short-Term Memory) networks. The primary objective is to predict the news category based on the text description. The notebook delves into data preprocessing, model creation, training, evaluation, and the visualization of results.

## Table of Contents

- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Saved Models](#saved-models)
- [Dependencies](#dependencies)
- [How to Run](#how-to-run)

## Dataset

The notebook employs the AG News dataset, which comprises news articles aggregated from over 2000 news sources. The dataset is categorized into 4 classes of news:
1. World
2. Sports
3. Business
4. Science/Technology

## Preprocessing

- Loaded data using TorchText's AG_NEWS dataset.
- The sentences were preprocessed by removing punctuation, converting to lowercase, and padding or marking unknown words.
- Words were converted to indices using a vocabulary derived from the GloVe embeddings.

## Model Architecture

The LSTM-based text classification model includes:
- **Embedding Layer**: Converts tokens into dense vectors and is initialized with GloVe embeddings.
- **LSTM Layer**: A Long Short-Term Memory layer that captures sequence data.
- **Fully Connected Layer**: Classifies the input sequence into one of the 4 news categories.

## Training

- Loss Function: Cross Entropy Loss
- Optimizer: Adam
- Device: CUDA (if available, else CPU)
- Different models were trained with varying embedding dimensions, and their performances were visualized using loss and accuracy plots.

## Evaluation

The trained models were assessed on a test dataset using accuracy as the primary metric. Evaluation also included the visualization of confusion matrices for different versions of the trained model ("models/model50.pt", "models/model100.pt", "models/model200.pt", "models/model300.pt").

## Saved Models

Trained models can be found in the `models` directory with names indicating their respective embedding dimensions, such as:
- model50.pt
- model100.pt
- model200.pt
- model300.pt

## Dependencies

- TorchText
- PyTorch
- numpy
- matplotlib
- seaborn

## How to Run

1. Clone the repository.
2. Install the required dependencies.
3. Open and execute the provided Jupyter notebook.
