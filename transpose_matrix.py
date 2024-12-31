def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
	if not a or not a[0] : 
		return []
	rows : int = len(a)
	cols : int = len(a[0])

	trans_a : list[list[int|float]]= [[0 for _ in range(rows)] for _ in range(cols)]

	for i in range(rows):
		for j in range(cols):
			trans_a[j][i] = a[i][j]
	
	return trans_a
