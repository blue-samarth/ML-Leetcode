import numpy as np

def adam_optimizer(parameter : float | int, grad : float, m : float, v : float, t : float,
				    learning_rate : float = 0.001, beta1 : float = 0.9, beta2 : float = 0.999,
					  epsilon : float =1e-8):

	m : float  = beta1 * m + (1 - beta1) * grad
	v : float = beta2 * v + (1 - beta2) * grad**2

	bias_corrected_m : float = m / (1 - beta1**t)
	bias_corrected_v : float = v / (1 - beta2**t)

	update : float = learning_rate * bias_corrected_m / np.sqrt(bias_corrected_v) + epsilon

	parameter : float | int = parameter - update

	return np.round(parameter , 5) , np.round(m , 5) , np.round(v , 5)