import numpy as np

# One hot encoding for nominal values
def to_categorical(x : np.ndarray , n_col : int = None) -> list[list[int | float]] :
    """
    This function performs one hot encoding of nominal values.
    If n_col is not provided, it will be automatically determined.
    Args:
        x : np.ndarray : The data.
        n_col : int : The number of columns.
    Returns:
        list[list[int | float]] : The one hot encoded data.
    """

    if not n_col:
        n_col = np.amax(x) + 1 # the maximum value of x + 1 is the number of columns because the index starts from 0 

    one_hot : np.ndarray = np.zeros((x.shape[0] , n_col))
    one_hot[np.arange(x.shape[0]) , x] = 1

    return one_hot.tolist()