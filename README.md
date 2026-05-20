# MNIST Classification with NumPy MLP and CNN

This repository contains my implementation for Project 1 of Neural Network and Deep Learning. The goal is to implement basic neural network components from scratch using NumPy and evaluate MLP and CNN models on MNIST.

## Implemented Components

The main implementations are in the following files:

- `mynn/op.py`
  - Linear layer forward and backward propagation
  - Softmax cross-entropy loss
  - 2D convolution operator
- `mynn/models.py`
  - MLP baseline model
  - CNN model
- `mynn/optimizer.py`
  - SGD optimizer
  - Momentum optimizer implementation
- `mynn/lr_scheduler.py`
  - Learning rate scheduler implementations

## Experiments

The experiments include:

1. MLP baseline on MNIST
2. CNN model and comparison with MLP
3. Learning rate comparison
4. CNN error analysis with confusion matrix and misclassified examples

The main scripts are:

- `test_train.py`: train MLP or CNN models
- `test_model.py`: evaluate a saved model
- `train_lr_experiment.py`: compare different learning rates
- `error_analysis.py`: generate confusion matrix and misclassified examples

## Results

The CNN model achieved a test accuracy of 98.18% on MNIST.

Additional figures are saved in the `figs/` folder, including the confusion matrix, misclassified examples, and learning rate comparison.

## Notes

The MNIST dataset and trained model checkpoints are not included in this repository. The trained checkpoint is provided separately in the project report.
