import matplotlib.pyplot as plt
import numpy as np
import h5py


f = h5py.File('patient-0-files-features-masks\patient_000_node_0.h5', 'r')
coords = f['coords']
features = f['features']

n_points = coords.shape[0]

X=features


def assign_clusters(X, centroids):
    distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
    
    # Assign each data point to the closest centroid
    clusters = np.argmin(distances, axis=0)
    
    return clusters

def update_centroids(X, clusters, k):
    new_centroids = np.zeros((k, X.shape[1]))
    for i in range(k):
        new_centroids[i] = np.mean(X[clusters == i], axis=0)
    return new_centroids




#Initialisation of clusters
k=5
indices = np.random.choice(n_points, k, replace=False)
indices.sort()
centroids = X[indices] #indices have to be sorted for h5



max_iter = 20
for _ in range(max_iter):
    print(_)
    clusters = assign_clusters(X, centroids)
    new_centroids = update_centroids(X, clusters, k)
    
    if np.all(centroids == new_centroids):
        break
    
    centroids = new_centroids
    

print(centroids)

plt.scatter(coords[:, 0], coords[:, 1], c=clusters, cmap='viridis', s=1, alpha=0.5)
plt.axis('image')
plt.show()