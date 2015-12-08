import numpy as np

a = np.zeros((1, 10))
b = np.ones((10, 5))
c = np.concatenate((a,b))
print a
print b
print c