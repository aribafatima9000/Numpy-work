import numpy as np

# 1. Array banana
arr1 = np.array([1, 2, 3, 4, 5])
print("Array:", arr1)

# 2. 2D Array (Matrix)
arr2 = np.array([[1, 2, 3], [4, 5, 6]])
print("\n2D Array:\n", arr2)

# 3. Zero aur Ones array
print("\nZeros:\n", np.zeros((2, 3)))
print("\nOnes:\n", np.ones((3, 3)))

# 4. Range banani (jaise Python me range hoti hai)
print("\nRange:", np.arange(1, 11))

# 5. Evenly spaced numbers
print("\nLinspace:", np.linspace(0, 1, 5))

# 6. Random numbers
print("\nRandom (0-1):\n", np.random.rand(2, 3))

# 7. Array ka shape, size
print("\nShape:", arr2.shape)
print("Size:", arr2.size)

# 8. Mathematical operations
arr3 = np.array([10, 20, 30, 40])
print("\nSum:", arr3.sum())
print("Mean:", arr3.mean())
print("Max:", arr3.max())
print("Min:", arr3.min())

# 9. Indexing & Slicing
print("\nFirst element:", arr3[0])
print("Last two elements:", arr3[-2:])

# 10. Reshape
arr4 = np.arange(1, 10)
print("\nOriginal:", arr4)
print("Reshaped (3x3):\n", arr4.reshape(3, 3))
