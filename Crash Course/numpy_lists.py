#numpy arrays have less memory allogations and are faster
#compared to lists

import numpy as np

array_two = np.arange(1, 4) ** 2
array_three = np.arange(1,4) ** 3

print(array_two)
print(array_three)

print(np.power(np.array([1,2,3]), 4))

print(np.negative(np.array([1,2,3])))

print(np.exp(np.array([1,2,3])))

print(np.log(np.array([1,2,3])))

print(np.sin(np.array([1,2,3])))
