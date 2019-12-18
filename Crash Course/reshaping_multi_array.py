import numpy as np

x = np.arange(9).reshape(3,3)
print(x)

#creates a view of the origina array. any modifications to
#ravel array causes change to the original array
ravelled_array = x.ravel()
print(ravelled_array)

#crates a copy of the original array
#any changes to the flatten array won't change the original array
flatten_array = x.flatten()
print(flatten_array)

y = np.arange(9)
y.shape = [3,3]
#transpose of an array
print(y.transpose())
print(y.T)

print('\n\n', np.resize(y, (6,6)))
print('\n\n', np.zeros((6,6), dtype = int))
print('\n\n', np.ones((6,6), dtype = int))
print('\n\n', np.eye((3), dtype=int))
print('\n\n', np.random.rand(3,3))
