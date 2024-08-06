import umap
import numpy as np
import matplotlib.pyplot as plt

import k_means_3

centers_vect = [0,1]
n_sane =2
n_unsane =2
coords, features, n_points_vect, patient_vect_id = k_means_3.load_data(centers_vect, n_sane, n_unsane)

n = len(n_points_vect)
X=[]
sum_np =0

for i in range(n):
    X.append (features[sum_np : sum_np+n_points_vect[i]])
    sum_np +=n_points_vect[i]

#inhomogeneous lenghts of the different vectors
max_length = max(len(x) for x in X)

# Pad the feature vectors with zeros to the maximum length
X_padded = np.array([np.pad(x, (0, max_length - len(x)), 'constant') for x in X])


# Initialize UMAP model
umap_model = umap.UMAP(n_components=2, random_state=42)

# Fit and transform the data
X_umap = umap_model.fit_transform(X_padded)

# Plotting the results
plt.figure(figsize=(10, 8))
plt.scatter(X_umap[:, 0], X_umap[:, 1], s=50, cmap='Spectral')
plt.title('UMAP projection of the feature vectors')
plt.xlabel('UMAP1')
plt.ylabel('UMAP2')
plt.colorbar()
plt.show()
