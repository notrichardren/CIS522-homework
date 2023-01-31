import numpy as np

# Chaithanya Sai Karne as a collaborator

class LinearRegression:

    w: np.ndarray
    b: float

    def __init__(self):
        self.w = None
        self.b = None

    def fit(self, X, y):  # analytical solution
        # params = np.linalg.inv(X.T @ X) @ X.T @ y
        # self.w = params[:-1]
        # self.b = params[-1]
        self.w = np.linalg.inv(X.T @ X) @ X.T @ y
        self.b = np.zeros(X.shape[0]) # NEED TO CHANGE THIS
        # supposed to be a float

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

        guo chei bryan jesse kabian lavnik yishak

        for i in range(epochs):

            # Make prediction
            y_hat = self.predict(X)

            # Compute loss as the mean squared error
            l = (y_hat - y) ** 2
            print(f"Epoch {i}: Loss {l}")

            # Compute gradient



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
