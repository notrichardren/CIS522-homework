import numpy as np


class LinearRegression:
    """
    A linear regression model that uses an analytical solution to fit the model.
    Bias not implemented because Ed says it was OK to do so.
    """

    w: np.ndarray
    b: float

    def __init__(self):
        self.w = np.zeros((2, 2))
        self.b = np.zeros((2, 2))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the model.

        Arguments:
            X (np.ndarray): The input data.
            Y (np.ndarray): The output data.

        Returns:
            None
        """

        if np.linalg.det(X.T @ X) != 0:
            self.w = (np.linalg.inv(X.T @ X)) @ (X.T @ y)
        else:
            print(
                "Closed form solution not possible here as determinant of X.T@X is not zero"
            )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """
        self.b = np.zeros((X.shape[0],))
        preds = (X @ self.w.T) + self.b
        return preds


class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression model that uses gradient descent to fit the model.
    Bias not implemented because Ed says it was OK to do so.
    """

    def fit(
        self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000
    ) -> None:
        """
        Fit the model.

        Arguments:
            X (np.ndarray): The input data.
            Y (np.ndarray): The output data.
            lr (float): The learning rate.
            epochs (int): The number of epochs.
        Returns:
            None
        """

        np.random.seed(42)
        self.w = np.random.randn(X.shape[1], 1)  # initial weights
        y = y.reshape(y.shape[0], 1)
        losses = []

        for i in range(0, epochs):
            preds = X @ self.w
            loss = (np.sum((preds - y) ** 2)) / X.shape[0]
            losses.append(loss)
            if i % 100 == 0:
                print(f"Loss in epoch {i} is {loss}")
            gradients = 2 * (X.T.dot(preds - y)) / X.shape[0]
            self.w = self.w - lr * gradients

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """
        preds = X @ self.w
        return preds
