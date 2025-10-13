import numpy as np

ff = np.load('features-filling-fld_000.npy')
print(f'Features shape: {ff.shape}')
maxima = np.max(ff, axis=0)
minima = np.min(ff, axis=0)

print('Unvariable columns:')
for n, (min, max) in enumerate(zip(minima, maxima)):
	if min == max:
		print(f'Column {n}, value {min}')

