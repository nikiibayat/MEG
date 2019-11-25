import hdf5storage
data = hdf5storage.loadmat('category_indices.mat')
print(data['scenes'])
