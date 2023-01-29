#%%

import numpy as np

#%%


class LinearRegression:

    w: np.ndarray
    b: float

    def __init__(self):
        self.w = None
        self.b = None

    def fit(self, X, y): # analytical solution
        w = np.linalg.inv(x.T @ X) @ X.T @ Y
        b = # no clue

    def predict(self, X):
        X*self.w+self.b


#%%

class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression model that uses gradient descent to fit the model.
    """

    def fit(
        self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000
    ) -> None:
        for i in range(epochs):
            
            # make predictions
            # compute loss


        raise NotImplementedError()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """
        return X*self.w+self.b