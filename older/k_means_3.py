import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import h5py
import os
import time
from tqdm import tqdm

import csv_reader
#import proto_show

def load_data(center_vect, n_sane, n_unsane):
    #Lets you choose how many sane and unsane are loaded in total (regardless of center)
    all_features = []
    all_coords = []
    n_points_vect =[]
    patient_id_vect =[]
    n_sane_counter =0
    n_unsane_counter =0

    for center in center_vect :
        #there are 20 patients per centers
        #i is the patient number
        for i in range (center*20,(center+1)*20):
            if n_sane_counter >=n_sane and n_unsane_counter >= n_unsane :
                break
                
            if csv_reader.is_sane(i) and n_sane_counter <n_sane :
                print(f"Loading patient {i} (sane)")
                for j in range(0,5): #adding the 5 nodes
                    filename = "C:\Edouard\Tecnico\Code\Train_features\center"+str(center)+"\patient_"+str(i).zfill(3)+"_node_"+str(j)+".h5"
                    if os.path.exists(filename):
                        with h5py.File(filename, 'r') as f:
                            coords = f['coords'][:]
                            features = f['features'][:]
                            all_coords.append(coords)
                            all_features.append(features)
                            n_points_vect.append(coords.shape[0])
                            
                    else :
                        print(f"There is no patient {i}, node {j}")
                        #I consider that ther is no patient with 0 node
                n_sane_counter +=1
                patient_id_vect.append(i)
            
            if (not csv_reader.is_sane(i)) and n_unsane_counter < n_unsane :
                print(f"Loading patient {i} (unsane)")
                for j in range(0,5): #adding the 5 nodes
                    filename = "C:\Edouard\Tecnico\Code\Train_features\center"+str(center)+"\patient_"+str(i).zfill(3)+"_node_"+str(j)+".h5"
                    if os.path.exists(filename):
                        with h5py.File(filename, 'r') as f:
                            coords = f['coords'][:]
                            features = f['features'][:]
                            all_coords.append(coords)
                            all_features.append(features)
                            n_points_vect.append(coords.shape[0])
                            
                    else :
                        print(f"There is no patient {i}, node {j}")
                        #I consider that ther is no patient with 0 nodes
                n_unsane_counter +=1
                patient_id_vect.append(i)
            
    return np.concatenate(all_coords), np.concatenate(all_features, axis=0), n_points_vect, patient_id_vect

def assign_clusters(X, centroids):
    distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
    clusters = np.argmin(distances, axis=0)
    return clusters

def update_centroids(X, clusters, k):
    new_centroids = np.zeros((k, X.shape[1]))
    for i in range(k):
        new_centroids[i] = np.mean(X[clusters == i], axis=0)
    return new_centroids

# lines = ['     ','o    ','oo   ','ooo  ','oooo ','ooooo']
# @animation.wait(lines, color="blue")
def k_means(k,max_iter, centers_vect, n_sane, n_unsane):
    coords, features, n_points_vect, patient_vect_id = load_data(centers_vect, n_sane, n_unsane)
    n_points_total = coords.shape[0]

    X = features

    # Initialisation of clusters
    indices = np.random.choice(n_points_total, k, replace=False)
    indices.sort()
    centroids = X[indices]  # indices have to be sorted for h5

    
    print(f"{max_iter} iteration of k-means, please wait")

    for _ in tqdm(range(max_iter)):
        clusters = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, clusters, k)
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    
    return coords, clusters, n_points_vect, patient_vect_id


def show_segmentation(patient, node, coords, clusters, n_points_vect, k) :
    title = f"Patient {patient} , Node "+str(node)
    plt.figure(title)
    plt.subplot(131)
    #Image sizes are different from one to another
    
    
    start = int(np.sum(n_points_vect[0:patient]))
    end = int(start + n_points_vect[patient])
    plt.scatter(coords[start:end, 0], coords[start:end, 1], c=clusters[start:end], cmap='viridis', s=1, alpha=0.5)
    plt.gca().invert_yaxis()
    plt.axis('image')
    plt.axis('off')

    plt.subplot(132)
    name = "C:\Edouard\Tecnico\Code\Train_features\stitches"+"\patient_"+str(patient).zfill(3)+"_node_"+str(node)+".jpg"
    img = mpimg.imread(name)
    plt.imshow(img)
    plt.axis('image')
    plt.axis('off')
    

    plt.subplot(133)
    plt.hist(clusters, bins=k, rwidth = 1.9, color ='skyblue', density =True, label = 'Dataset distribution')
    plt.hist(clusters[start:end], bins=k, rwidth=0.6, color='darkblue', density =True, label = 'Sane distribution')
    plt.legend()
    plt.show()

def make_histo(clusters,k, n_point_vect,patient_id_vect, set_of_sane):
    hist_sane = np.zeros(k)
    hist_unhealthy = np.zeros(k)
    node_count =0
    patch_id =0
    for patient in patient_id_vect : #a patient is counted as many times as it has nodes
        
        if patient in set_of_sane :
            
            for i in range(n_point_vect[node_count]):
                hist_sane[clusters[patch_id]]+=1
                patch_id +=1
                
        else :
            for i in range(n_point_vect[node_count]):
                hist_unhealthy[clusters[patch_id]]+=1
                patch_id +=1
        node_count+=1
    
    return hist_sane, hist_unhealthy

def draw_histograms(hist_sane, hist_unhealthy, n_sane, n_unhealthy):

    #this mean corrects the desequilibrium between n_sane and n_unhealthy
    hist_mean = (hist_sane * n_unhealthy + hist_unhealthy * n_sane) / (n_sane+ n_unhealthy)
    
    k = len(hist_mean)
    indices = np.arange(k)
    
    # Définir la largeur des barres
    width = 0.3

    # Tracer l'histogramme moyen en arrière-plan en gris
    plt.bar(indices, hist_mean, width=1.0, color='lightgray', label='Mean Histogram', zorder=1)
    
    # Tracer l'histogramme des patients sains légèrement décalé à gauche en vert
    plt.bar(indices - width / 2, hist_sane, width=width, color='green', alpha=0.6, label='Sane Histogram', zorder=2)
    
    # Tracer l'histogramme des patients malades légèrement décalé à droite en rouge
    plt.bar(indices + width / 2, hist_unhealthy, width=width, color='red', alpha=0.6, label='Unhealthy Histogram', zorder=2)
    
    # Ajouter des étiquettes et un titre
    plt.xlabel('Cluster Index')
    plt.ylabel('Frequency')
    plt.title('Cluster Histograms')
    plt.legend()
    plt.show()

def main():
    t0 = time.time()
    k=5
    max_iter =2
    n_sane =2
    n_unsane =2
    centers =[0,1]
    coords, clusters, n_points_vect, patient_id_vect = k_means(k,max_iter, centers, n_sane, n_unsane)
    t1 =time.time()
    print(f"Execution in {t1-t0} seconds")
    print(f"(or {(t1 -t0)/60} minutes)")
    
    set_of_sane = csv_reader.set_of_sanes()
    hist_sane, hist_unhealthy = make_histo(clusters,k, n_points_vect,patient_id_vect, set_of_sane)
    print(hist_sane, hist_unhealthy)
    draw_histograms(hist_sane, hist_unhealthy, n_sane, n_unsane)
    #show_segmentation(0,0, coords, clusters, n_points_vect, k)
    # patient=0,node=0
    

if __name__ == "__main__":
    main()


