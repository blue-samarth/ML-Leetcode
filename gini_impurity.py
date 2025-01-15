import numpy as np

# Gini impurity for classification
#  Author : Samarth Sharma
# Date : 2021-10-06

def gini_impurity(y : np.ndarray | list[int | float] ) -> float:
    """Calculate the Gini impurity of a dataset.
    
    Gini impurity measures how often a randomly chosen element would be incorrectly classified
    if it were randomly labeled according to the distribution of labels in the subset.
    It reaches its minimum (0.0) when all cases in the node fall into a single target category.
    A perfect splitting produces a Gini gain of 0.5.
    
    Args:
        y: Array-like object containing target values. Can be either a numpy array
           or a list of numerical values (int or float). 
    Returns:
        float: Gini impurity score rounded to 3 decimal places.
    Raises:
        ValueError: If input is empty or contains non-numeric values.
    """


    n : int = len(y)

    if n == 0: raise ValueError("Input array is empty")
    y_arr : np.ndarray = np.array(y)
    
    if not np.issubdtype(y_arr.dtype, np.number):
        raise ValueError("Input array must contain only numeric values (int or float)")
        
    counts : np.ndarray = np.unique(y_arr, return_counts=True)[1]
    
    # Calculate probabilities and sum their squares
    probabilities : np.ndarray = counts / n
    gini : float = 1 - np.sum(probabilities**2)
    
    return round(gini, 3)