def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
	"""
	This function calculates the mean of the matrix based on the mode.
	"""
	mean : float = 0.0
	rows : int = len(matrix)
	columns : int = len(matrix[0])
	res  : list[float] = []

	if mode.lower() == 'column':
		for c in range(columns):
			mean = 0.0
			for r in range(rows):
				mean += matrix[r][c]/columns
			res.append(mean)
	elif mode.lower() == 'row':
		for r in range(rows):
			mean = 0.0
			for c in range(columns):
				mean += matrix[r][c]/rows
			res.append(mean)
	else:
		raise ValueError("Invalid mode. Please use 'column' or 'row'")
	return res

print()