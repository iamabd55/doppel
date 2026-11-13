B=8
cells = [(i, j) for i in range(B) for j in range(B)]
print(cells)
sorted_cells=sorted(cells, key=lambda p: (p[0] + p[1],p[0] if (p[0] + p[1]) % 2 else -p[0]))
print(sorted_cells[:16])