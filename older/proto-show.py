import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import h5py
import os
import cv2


def load_data_from_folder(folder_path,nb_max_img):
    all_features = []
    all_coords = []
    n_points_vect =[]
    n=0
    for filename in os.listdir(folder_path):
        if n < nb_max_img :
            if filename.endswith('.h5'):
                file_path = os.path.join(folder_path, filename)
                with h5py.File(file_path, 'r') as f:
                    coords = f['coords'][:]
                    features = f['features'][:]
                    all_coords.append(coords)
                    all_features.append(features)
                    n_points_vect.append(coords.shape[0])
            n+=1
        else :
            break

    return np.concatenate(all_coords), np.concatenate(all_features, axis=0), n_points_vect

def assign_clusters(X, centroids):
    distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
    clusters = np.argmin(distances, axis=0)
    return clusters

def update_centroids(X, clusters, k):
    new_centroids = np.zeros((k, X.shape[1]))
    for i in range(k):
        new_centroids[i] = np.mean(X[clusters == i], axis=0)
    return new_centroids

# Load data from folder containing multiple images
folder_path = 'C:\Edouard\Tecnico\Code\Train_features\center0'
coords, features, n_points_vect = load_data_from_folder(folder_path,5)
n_points_total = coords.shape[0]

X = features

# Initialisation of clusters
k = 5
indices = np.random.choice(n_points_total, k, replace=False)
indices.sort()
centroids = X[indices]  # indices have to be sorted for h5

max_iter = 2
for _ in range(max_iter):
    print(_)
    clusters = assign_clusters(X, centroids)
    new_centroids = update_centroids(X, clusters, k)

    if np.all(centroids == new_centroids):
        break

    centroids = new_centroids



i = 0 #just for one image
"""title = 'Patient 0, Node '+str(i)
plt.figure(title)
plt.subplot(131)
#Image sizes are different from one to another
start = int(np.sum(n_points_vect[0:i]))
end = int(start + n_points_vect[i])
plt.scatter(coords[start:end, 0], coords[start:end, 1], c=clusters[start:end], cmap='viridis', s=1, alpha=0.5)
plt.gca().invert_yaxis()
plt.axis('image')
plt.axis('off')

plt.subplot(132)
name = "C:\Edouard\Tecnico\Code\patient-0-files-features-masks"+"\patient_000_node_"+str(i)+".jpg"
img = mpimg.imread(name)
plt.imshow(img)
plt.axis('image')
plt.axis('off')


plt.subplot(133)
plt.hist(clusters, bins=k, rwidth = 1.9, color ='skyblue', density =True, label = 'Dataset distribution')
plt.hist(clusters[start:end], bins=k, rwidth=0.6, color='darkblue', density =True, label = 'Image distribution')
plt.legend()
plt.show()"""



name = "C:\Edouard\Tecnico\Code\Train_features\stitches"+"\patient_000_node_"+str(i)+".jpg"
image = plt.imread(name)
print(image.shape)

plt.figure()
for ki in range(k):
    ntile = 20
    x1 = coords[ntile,0]//64
    y1 = coords[ntile,1]//64
    x2 = x1 +256//64
    y2 = y1 +256//64

    selection = image[y1:y2, x1:x2, :]
    while clusters[ntile] != ki or selection.sum() < 4000:
        ntile+=1
        
        x1 = coords[ntile,0]//64
        y1 = coords[ntile,1]//64
        x2 = x1 +256//64
        y2 = y1 +256//64
        selection = image[y1:y2, x1:x2, :]
    print(x1,x2)
    print(y1,y2)
    print(selection)
    label = "Proto of cluster "+str(ki)
    print(label)
    
    plt.subplot(230+ki+1)
    plt.xlabel(label)
    plt.imshow(selection)
    #plt.axis('off')
    plt.axis('image')
    ntile =0

plt.show()

plt.figure()
plt.imshow(image)
plt.scatter([y1,y2],[x1,x2],s=20)
plt.show()
