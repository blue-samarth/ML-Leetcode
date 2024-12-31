def matrix_dot_vector(a:list[list[int|float]],b:list[int|float])-> list[int|float] | int:
	row_b : int = len(b)
	col_a : int = len(a[0])
	res : list = []
	# since matrix can be multiplied when col_a == row_b
	if col_a != row_b: 
		return -1
	for row in a:
		cur : int|float = 0
		for j in range(col_a):
			cur += row[j] * b[j]
		res.append(cur)
	return res