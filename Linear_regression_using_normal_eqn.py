import numpy as np

def linear_regression_normal_eqn(X : list[list[float]] , y : list[float]) -> np.ndarray:
    X : np.ndarray = np.array(X)
    y : np.ndarray = np.array(y)
    
    # Normal Equation
    # theta = (X^T * X)^-1 * X^T * y
    # where X is the feature matrix and y is the target vector


    theta : np.ndarray = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    
    return theta
    