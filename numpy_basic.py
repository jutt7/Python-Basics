import numpy as np

a = np.array([[1, 2, 3], [15, 20, 3.2]])
print(a.ndim)
print(a.shape)


output = np.ones((5, 5))
print(output)

z = np.zeros((3, 3))

z[1, 1] = 9

print(z)

output[1:4, 1:4] = z

print(output)
