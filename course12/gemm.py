import numpy as np

img = np.array([[1, 2, 4, 5], [2, 3, 5, 6], [4, 5, 7, 8], [5, 6, 8, 9]])
kernel = np.array([1, 2, 3, 4])


out = img @ kernel
print(out)
