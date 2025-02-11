import numpy as np
import matplotlib.pyplot as plt

def compute_error_for_line_given_points(b, m, points):
    totalError = 0
    for i in range(len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m * x + b)) ** 2
    return totalError / float(len(points))

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m
    errors = []

    for i in range(num_iterations):
        b, m = step_gradient(b, m, np.array(points), learning_rate)
        errors.append(compute_error_for_line_given_points(b, m, points))

    return b, m, errors

def step_gradient(b_current, m_current, points, learning_rate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))

    for i in range(len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))

    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)

    return new_b, new_m

def plot_regression_line(points, b, m, title):
    x_vals = np.array(points[:, 0])
    y_vals = np.array(points[:, 1])
    
    plt.scatter(x_vals, y_vals, color="blue", label="Data Points")
    plt.plot(x_vals, m * x_vals + b, color="red", label="Regression Line")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)
    plt.legend()
    plt.show()

def plot_error_reduction(errors):
    plt.plot(range(len(errors)), errors, color="green")
    plt.xlabel("Iterations")
    plt.ylabel("Error")
    plt.title("Error Reduction over Iterations")
    plt.show()

def run():
    points = np.genfromtxt('data.csv', delimiter=',')

    learning_rate = 0.0001
    initial_b = 0
    initial_m = 0
    num_iterations = 1000

    # Plot initial regression line
    plot_regression_line(points, initial_b, initial_m, "Initial Regression Line")

    print(f"Starting gradient descent at b = {initial_b}, m = {initial_m}, error = {compute_error_for_line_given_points(initial_b, initial_m, points)}")
    
    b, m, errors = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)

    print(f"Ending point at b = {b}, m = {m}, error = {compute_error_for_line_given_points(b, m, points)}")

    # Plot final regression line
    plot_regression_line(points, b, m, "Final Regression Line After Gradient Descent")

    # Plot error reduction over iterations
    plot_error_reduction(errors)

if __name__ == '__main__':
    run()
