import numpy as np
from sklearn.cluster import MiniBatchKMeans


def k_means(k, max_iter, n_init, features):
    print("Initializing the model")
    kmeans = MiniBatchKMeans(n_clusters=k, max_iter=max_iter, n_init=n_init, batch_size=100, random_state=42)

    print(f"{max_iter} iteration(s) of MiniBatchKMeans, please wait")
    clusters = kmeans.fit_predict(features)
    centroids = kmeans.cluster_centers_

    return kmeans, clusters, centroids

def cluster_from_centroids(kmeans_model, new_features):
    print("Using the clusters for the test data")
    new_clusters = kmeans_model.predict(new_features)
    return new_clusters

def make_histo_image_level(clusters, k, n_point_vect):
    
    n_images = len(n_point_vect) 
    #n_point_vect gives the number of patch per image

    hist_matrix = np.zeros((n_images, k) )

    patch_id = 0
    for i in range(n_images) :
        n_patch = n_point_vect[i]
        for j in range(n_patch):
            hist_matrix[i,clusters[patch_id]]+=1
            patch_id +=1

    #normalisation of histograms
    normalized_matrix = hist_matrix * (100 / hist_matrix.sum(axis=1, keepdims=True))
    return normalized_matrix

def main():
    return 0

if __name__ == '__main__':
    main()