# Linear Regression from Scratch

# Overview

This project implements Linear Regression using Gradient Descent in Python to fit a line to data while minimizing Mean Squared Error (MSE).

# Features

Batch Gradient Descent for optimization

MSE Calculation for model evaluation

Data & Regression Line Visualization

Error Reduction Graph

# Installation

Install dependencies:

pip install numpy matplotlib

# Usage

1. Prepare Data: Place dataset in data.csv in format:

X, Y
1, 2
2, 2.8
3, 3.6

2. Run Script:

python linear-regression.py

3. Output:

Initial & Final Regression Line

Error Reduction Graph

Optimized m & b values

# Key Functions

compute_error_for_line_given_points(b, m, points): Calculates MSE.

step_gradient(b, m, points, learning_rate): Computes and updates gradients.

gradient_descent_runner(points, b, m, learning_rate, num_iterations): Iteratively optimizes m and b.

plot_regression_line(points, b, m, title): Visualizes regression line.

plot_error_reduction(errors): Shows error reduction over iterations.

# License

Open-source and free to use.

# Contact

For questions, reach out!