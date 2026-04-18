# Hangman Solver - Bi-LSTM

Solving the game of Hangman using a 6-layer Bidirectional LSTM neural network, 
trained on ~250,000 dictionary words and tested on a completely disjoint set of 1,000 unseen words.

## Approach

The model takes the current masked word (e.g. `_ p p _ e`) as input and predicts 
the most likely next letter. 

Architecture:
- 6 Bidirectional LSTM layers
- Dropout layer for regularization
- Fully connected output layer for letter classification

The model learns letter co-occurrence patterns and positional context from both 
directions of the word, significantly outperforming n-gram and frequency-based baselines.

## Results

| Method | Accuracy |
|--------|----------|
| Benchmark (frequency baseline) | ~18% |
| **Bi-LSTM (this work)** | **61.6%** |

## Tech Stack
Python · PyTorch · NumPy

## References
- CS224N: Natural Language Processing with Deep Learning (Stanford)
