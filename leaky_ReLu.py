def leaky_relu(z : float | int , alpha : float = 0.01) -> float | int :
    # return max(alpha*z, z) This would be a bit slower
    return z if z > 0 else alpha*z # This is relatively faster
    # return np.maximum(alpha*z, z) # This is the fastest