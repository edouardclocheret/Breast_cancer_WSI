import h5py

f = h5py.File('patient-0-files-features-masks\patient_000_node_0.h5', 'r')


coords = f['coords']
features = f['features']

n_points = coords.shape[0]

"""
print(list(f.keys()))
print(coords.dtype, coords.shape)
print(features.dtype, features.shape)
print(coords[20],features[20])
"""



