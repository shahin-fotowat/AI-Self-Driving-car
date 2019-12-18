import numpy as np

x = np.arange(3)
y = np.arange(3)
z = np.arange(3)

multi_array = np.array([x, y, z])
print(multi_array)
print(multi_array.shape)

w = np.linspace(1,10,50)
print(w)

v = np.linspace(1,100,3, False)
t = np.arange(1,100,3)

print(v)
print(t)
