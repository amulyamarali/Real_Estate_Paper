import numpy as np
import faiss

# Generate sample data (replace this with your actual data)
n = 1000  # Number of data points
d = 64    # Dimensionality of the data

data = np.random.random((n, d)).astype('float32')

# Build an HNSW index
index = faiss.IndexHNSWFlat(d, 32)  # 32 is the number of neighbors for each data point

# Add data to the index
index.add(data)

# Perform a query to find nearest neighbors
k = 5  # Number of nearest neighbors to retrieve
query_data = np.random.random((1, d)).astype('float32')  # Replace with your query data

D, I = index.search(query_data, k)

# D contains distances, I contains indices of the nearest neighbors
print("Nearest neighbors indices:", I)
print("Distances to nearest neighbors:", D)
