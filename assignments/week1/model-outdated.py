import numpy as np


class LinearRegression:
    w: np.ndarray
    b: float

    def __init__(self):
        self.w = None
        self.b = None

    def fit(self, X, y):  # analytical solution
        self.w = np.linalg.inv(X.T @ X) @ X.T @ y
        self.b = ...
        # mean of difference -- Y - Wx = B. Average over every point.

    def predict(self, X):
        return X @ self.w


class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression model that uses gradient descent to fit the model.
    """

    def fit(
        self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000
    ) -> None:
        # Xavier initialization, n_in = 1, n_out = 1
        self.w = np.random.uniform(-np.sqrt(3), np.sqrt(3))
        self.b = np.random.uniform(-np.sqrt(3), np.sqrt(3))

        for i in range(
            epochs
        ):  # One epoch is one round on the entire dataset -- so just use the entire matrix
            # Make prediction
            y_hat = self.predict(X)

            # Compute loss as the mean squared error
            loss = (y_hat - y) ** 2
            print(f"Epoch {i}: Loss {loss}")

            # Compute gradient of the weight
            # Compute gradient of the bias

            # Update parameters
            self.w = ...
            self.b = ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """
        return X * self.w + self.b
