# Toxic Comment Classification using Bi-LSTM

This project implements a Bidirectional LSTM (Bi-LSTM) model for classifying toxic comments across multiple categories.

## Features

- Multi-label classification (6 toxicity categories)
- Text preprocessing with cleaning, lemmatization, and sentiment analysis
- Bi-LSTM model with optional attention mechanism
- Class weighting for imbalanced data
- REST API for real-time classification
- Batch processing capabilities

## Dataset

The model expects a CSV file with the following columns:
- `comment_text`: The text of the comment
- `toxic`: Binary label (1 if toxic, 0 otherwise)
- `severe_toxic`: Binary label
- `obscene`: Binary label
- `threat`: Binary label
- `insult`: Binary label
- `identity_hate`: Binary label
