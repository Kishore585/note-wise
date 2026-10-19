"""
Utility script to generate realistic sample lecture notes PDF for testing NoteWise.
"""

import os
import pymupdf


def create_sample_pdf():
    os.makedirs("sample_data", exist_ok=True)
    pdf_path = "sample_data/machine_learning_notes.pdf"

    doc = pymupdf.open()

    # Page 1: Introduction to Neural Networks & Optimization
    page1 = doc.new_page(width=595, height=842) # A4
    text_page1 = """LECTURE 04: NEURAL NETWORKS & GRADIENT DESCENT OPTIMIZATION
Instructor: Prof. Jennifer Vance | Department of Computer Science
Course: CS 482 - Deep Learning & Statistical Learning Theory

1. Introduction to Artificial Neural Networks (ANNs)
Artificial Neural Networks are computational models inspired by biological neural networks in the human brain. An ANN is composed of interconnected units called artificial neurons or perceptrons. Each neuron receives one or more input signals, computes a weighted sum of the inputs plus a bias term, and applies a non-linear activation function to produce an output signal.

The mathematical formulation for a single neuron is given by:
z = (w_1 * x_1) + (w_2 * x_2) + ... + (w_n * x_n) + b = (W^T * X) + b
a = sigma(z)

Where:
- W represents the weight vector representing synaptic connection strengths.
- X represents the input feature vector.
- b represents the scalar bias term shifting the activation threshold.
- sigma represents the non-linear activation function (such as ReLU, Sigmoid, or Tanh).

2. The Role of Non-Linear Activation Functions
Without non-linear activation functions, a multi-layer neural network collapses into a single linear transformation, regardless of how many hidden layers are stacked:
W_2 * (W_1 * X) = (W_2 * W_1) * X = W_combined * X

Common activation functions include:
- Rectified Linear Unit (ReLU): f(x) = max(0, x). ReLU mitigates the vanishing gradient problem in deep architectures, though it suffers from the dying ReLU phenomenon when neurons become inactive.
- Sigmoid Function: f(x) = 1 / (1 + exp(-x)). Maps inputs to the range (0, 1), often used in binary classification output layers.
- Hyperbolic Tangent (Tanh): f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x)). Zero-centered with output range (-1, 1).
"""
    page1.insert_text((40, 50), text_page1, fontsize=10.5, fontname="helv")

    # Page 2: Loss Functions, Backpropagation & Gradient Descent
    page2 = doc.new_page(width=595, height=842)
    text_page2 = """CS 482: LECTURE 04 (CONTINUED) - LOSS FUNCTIONS & BACKPROPAGATION

3. Loss Functions and Empirical Risk Minimization
A loss function measures the discrepancy between the network's predicted output (y_hat) and the ground truth label (y). The objective of training is to find the set of optimal weights W* that minimizes the cumulative empirical loss across the training dataset:

- Mean Squared Error (MSE): Used predominantly for regression tasks.
  L_MSE = (1 / (2 * N)) * sum((y_i - y_hat_i)^2)

- Categorical Cross-Entropy Loss: Used for multi-class classification problems with Softmax output.
  L_CE = - sum(y_i * log(y_hat_i))

4. The Backpropagation Algorithm
Backpropagation is the foundational algorithm used to calculate the gradient of the loss function with respect to every individual parameter (weights and biases) in the network. Backpropagation is an efficient application of the mathematical chain rule of calculus.

The chain rule dictates that the gradient of the loss L with respect to an internal weight w_ij is:
dL / dw_ij = (dL / da_j) * (da_j / dz_j) * (dz_j / dw_ij)

By caching intermediate activation values during the forward pass, backpropagation propagates error derivatives backward from the output layer to the earliest input layer in O(V + E) computational time, making deep learning computationally feasible.

5. Gradient Descent Optimizers
Once gradients are computed, an optimization algorithm updates the parameters in the direction of steepest descent:
W_new = W_old - (eta * dL / dW)

Where eta represents the learning rate (step size):
- Stochastic Gradient Descent (SGD): Updates weights using a single training sample at a time, introducing high variance and noisy updates.
- Mini-Batch Gradient Descent: Evaluates gradients across a mini-batch of k samples (e.g., 32, 64, 128), balancing computational efficiency and vector parallelization.
- Adam Optimizer (Adaptive Moment Estimation): Combines the benefits of momentum (exponential moving average of past gradients) and RMSProp (adapting per-parameter learning rates based on squared gradients).
"""
    page2.insert_text((40, 50), text_page2, fontsize=10.5, fontname="helv")

    # Page 3: Overfitting, Regularization & Exam Review
    page3 = doc.new_page(width=595, height=842)
    text_page3 = """CS 482: LECTURE 04 (CONTINUED) - REGULARIZATION & SUMMARY

6. Overfitting and Regularization Techniques
Overfitting occurs when a neural network memorizes noise in the training set rather than learning generalizable underlying patterns. An overfitted model exhibits low training loss but high validation error.

Standard regularization strategies include:
- L2 Regularization (Weight Decay): Adds a penalty proportional to the sum of squared weights to the loss function:
  L_total = L_data + (lambda / 2) * sum(w_i^2)
  This discourages excessively large weights and encourages smoother decision boundaries.
- Dropout: During each training iteration, randomly deactivates a fraction p (e.g., p = 0.5) of hidden neurons. This prevents complex co-adaptations between neurons and forces redundant representations.
- Early Stopping: Halts training when validation loss stops decreasing and begins to increase, preventing over-training.

7. Key Exam Takeaways
- Understand why ReLU replaced Sigmoid in hidden layers (addressing vanishing gradients).
- Know the exact mathematical steps of the forward pass and backward chain rule.
- Differentiate between Batch Gradient Descent, Mini-batch SGD, and Adam optimization.
- Be able to derive the gradient of Mean Squared Error with respect to linear weights.
- Always check that your learning rate is neither too large (divergence) nor too small (premature stagnation).
"""
    page3.insert_text((40, 50), text_page3, fontsize=10.5, fontname="helv")

    doc.save(pdf_path)
    doc.close()
    print(f"Sample PDF successfully created at {pdf_path}")


if __name__ == "__main__":
    create_sample_pdf()
