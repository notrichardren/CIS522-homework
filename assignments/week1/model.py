#%%

import numpy as np

#%%


class LinearRegression:

    w: np.ndarray
    b: float

    def __init__(self):  # Xavier initialization, n_in = 1, n_out = 1
        self.w = np.random.uniform(-np.sqrt(3), np.sqrt(3))
        self.b = np.random.uniform(-np.sqrt(3), np.sqrt(3))

    def fit(self, X, y):  # analytical solution
        params = np.linalg.inv(x.T @ X) @ X.T @ Y
        w = params[:-1]
        b = params[-1]

    def predict(self, X):
        X * self.w + self.b


#%%


class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression model that uses gradient descent to fit the model.
    """

    def fit(
        self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000
    ) -> None:

        # shuffle data

        for i in range(epochs):

            # do i need to use batches?

            # make prediction
            y_hat = self.predict(X)
            l = np.abs(y_hat - y)

            # make predictions
            # compute loss

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """
        return X * self.w + self.b
