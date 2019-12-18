import numpy as np

x = np.arange(9).reshape(3,3)
print(x)

y = np.arange(18).reshape(3,2,3)
print('\n', y)

print('\n\n', y[1, 0:2, 0:3])
#or we could just say
print('\n\n', y[1, ...])
