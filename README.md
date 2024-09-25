# Banishing the Vanishing Gradient

## Course: Pattern Recognition

## Overview

This project aims to address the vanishing gradient problem in neural networks through the implementation and analysis of two distinct architectures utilizing skip layers (also known as residual blocks). The effectiveness of these architectures is evaluated using the CIFAR-10 dataset, a widely used benchmark for image classification tasks.

## Objectives

- Implement two separate neural network architectures:
  1. A baseline architecture without skip connections.
  2. An enhanced architecture featuring skip layers/residual blocks.
  
- Analyze the performance of both architectures to highlight the impact of skip connections on training efficiency and accuracy.

- Demonstrate the effectiveness of skip layers in mitigating the vanishing gradient problem, which is a significant challenge in training deep neural networks.

## Files Included

- **Implementation Notebook (`473_Final_BVG.ipynb`)**: This Jupyter notebook contains the code for building, training, and evaluating the neural network architectures. It includes detailed explanations, visualizations of model performance, and insights into the training process.

- **Accuracy Calculation Script (`accuracy.py`)**: A Python script dedicated to calculating and outputting the accuracy of the trained models on the CIFAR-10 dataset. This script can be executed independently to retrieve the accuracy metrics after the models have been trained.

## Usage

To run this project, ensure you have the following dependencies installed:

- TensorFlow
- NumPy
- Matplotlib
- Pandas

You can install these dependencies via pip:

```bash
pip install tensorflow numpy matplotlib pandas

**Contributors:**
- Raymand Shojaie Aghabalaghe
- Thomas Peschlow
- Santiago Buitrago
