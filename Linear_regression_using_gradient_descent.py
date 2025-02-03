import numpy as np

def linear_regression_gradient_descent(X: np.ndarray, y: np.ndarray,
                                       alpha: float, iterations: int
                                       ) -> np.ndarray:
    """
    Performs linear regression using gradient descent to find optimal parameters.

    Args:
        X: np.ndarray = Feature matrix
        y: np.ndarray = Target values vector containing the output variable
        alpha: float = Learning rate that controls the step size of gradient descent
        iterations: int = Number of iterations to run gradient descent

    Returns:
        theta: np.ndarray = Optimized parameter vector (theta) rounded to 4 decimal places, 
        where each element represents the coefficient for a feature

    """
    m, n = X.shape
    theta = np.zeros((n, 1))

    for _ in range(iterations):
        predictions: np.ndarray = X@theta
        errors: np.ndarray = predictions - y.reshape(-1, 1)
        updates: np.ndarray = X.T @ errors/m
        theta -= alpha * updates
    return np.round(theta.flatten(), 4)
