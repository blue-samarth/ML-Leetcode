import numpy as np

def reshape_matrix(a: list[list[int|float]], rows: int, cols: int) -> list[list[int|float]]:
    arr = np.array(a)
    return arr.reshape(rows, cols)
