# Tune_function.py
# DESCRIPTION:
# This script provides an implementation of the MouseInputEstimator, which applies an Exponential Moving Average (EMA) filter to mouse input data. It is designed for smoothing cursor movement and reducing noise, especially in applications involving gaze tracking or dynamic input.

# FEATURES:
# Exponential Moving Average (EMA): Smooths mouse inputs to reduce jitter and improve tracking precision.
# Error Analysis: Computes the error between raw inputs and filtered outputs to evaluate performance.
# Adjustable Parameters: Includes configurable n (buffer size) and alpha (smoothing factor).
# How to Use:
# Initialize MouseInputEstimator with the desired buffer size (n) and smoothing factor (alpha).
# Add mouse or estimation inputs using add_mouse_input and add_estimation_input.
# Retrieve the current EMA-filtered values using get_current_ema.
# Key Functions:
# add_mouse_input: Adds raw mouse input coordinates to the buffer.
# add_estimation_input: Computes the EMA for new estimation inputs.
# get_current_ema: Returns the latest EMA-filtered coordinates.
# get_error: Calculates errors between raw inputs and filtered outputs.

import numpy as np

class MouseInputEstimator:
    def __init__(self, n=10, alpha=0.5):
        """
        Initialize the estimator with past n inputs and EMA alpha.

        :param n: Number of past inputs to store
        :param alpha: Smoothing factor for EMA (0 < alpha <= 1)
        """
        self.n = n
        self.alpha = alpha
        self.mouse_inputs = []  # List to store past mouse inputs
        self.estimation_inputs = []  # List to store past estimation inputs
        self.filtered_inputs = []  # EMA-filtered values

    def add_mouse_input(self, x, y):
        """
        Add a new mouse input.

        :param x: Mouse input X-coordinate
        :param y: Mouse input Y-coordinate
        """
        if len(self.mouse_inputs) >= self.n:
            self.mouse_inputs.pop(0)  # Remove oldest input if limit is reached
        self.mouse_inputs.append((x, y))

    def add_estimation_input(self, ex, ey):
        """
        Add a new estimation input and compute EMA.

        :param ex: Estimation input X-coordinate
        :param ey: Estimation input Y-coordinate
        """
        if len(self.estimation_inputs) >= self.n:
            self.estimation_inputs.pop(0)  # Remove oldest input if limit is reached
        self.estimation_inputs.append((ex, ey))

        # Compute EMA
        if len(self.filtered_inputs) == 0:
            # Initialize EMA with the first estimation input
            self.filtered_inputs.append((ex, ey))
        else:
            last_ema = self.filtered_inputs[-1]
            new_ema_x = self.alpha * ex + (1 - self.alpha) * last_ema[0]
            new_ema_y = self.alpha * ey + (1 - self.alpha) * last_ema[1]
            self.filtered_inputs.append((new_ema_x, new_ema_y))

        # Maintain EMA buffer size
        if len(self.filtered_inputs) > self.n:
            self.filtered_inputs.pop(0)

    def get_current_ema(self):
        """
        Get the current EMA-filtered value.

        :return: Tuple (filtered_x, filtered_y)
        """
        if len(self.filtered_inputs) > 0:
            new = self.filtered_inputs[-1]
            return self.filtered_inputs
        return None

    def get_error(self):
        """
        Calculate the error between mouse inputs and EMA-filtered estimations.

        :return: List of error values for the stored inputs
        """
        errors = []
        for (mx, my), (fx, fy) in zip(self.mouse_inputs, self.filtered_inputs):
            error = np.sqrt((mx - fx)**2 + (my - fy)**2)  # Euclidean distance
            errors.append(error)
        return errors


# Example Usage
if __name__ != "__main__":
    estimator = MouseInputEstimator(n=5, alpha=0.3)

    # Simulate mouse and estimation inputs
    mouse_data = [(10, 20), (12, 24), (14, 28), (16, 32), (18, 36)]
    estimation_data = [(11, 21), (13, 25), (15, 29), (17, 33), (19, 37)]

    for (mx, my), (ex, ey) in zip(mouse_data, estimation_data):
        estimator.add_mouse_input(mx, my)
        estimator.add_estimation_input(ex, ey)
        print("Current EMA:", estimator.get_current_ema())

    print("Errors:", estimator.get_error())
