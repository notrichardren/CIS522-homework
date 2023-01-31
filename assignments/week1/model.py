import numpy as np


class LinearRegression:

    w: np.ndarray
    b: float

    def __init__(self):
        self.w = None
        self.b = None

    def fit(self, X, y):  # analytical solution
        self.w = np.linalg.inv(X.T @ X) @ X.T @ y
        self.b = np.zeros(X.shape[0]) # NEED TO CHANGE THIS

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

        # Shuffle data

        # Do I need to use batches?
        y = y.reshape(y.shape[0], 1)

        for i in range(epochs):

            # Make prediction
            y_hat = self.predict(X)

            # Compute loss as the mean squared error
            l = (y_hat - y) ** 2
            print(f"Epoch {i}: Loss {l}")

            # Compute gradient


            # 2* (X.T.dot (preds- y) )/X. shape[ 0 ]

            # Update parameters
            self.w = 

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """
        return X * self.w + self.b
