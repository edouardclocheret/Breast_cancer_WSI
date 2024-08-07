import k_means_batch
import predictor
import file_manager
import network

import platform
import numpy as np


def param_choice():
    
    #prameters of clustering
    k = 3
    max_iter = 20
    n_init = 3 #number of iterations before updating centroids
    
    #Can be redefined if needed:
    centers_train = [0]
    centers_test = [2]

    return k, max_iter, centers_train, centers_test, n_init

def circular_train_test(k, max_iter, n_init, features_dir, path_to_csv):
    for i in range(5):
        centers_train = np.arange(5)
        centers_train = centers_train[centers_train != i]
        centers_test = [i]
        do_simulation_svm(centers_train, centers_test, k, max_iter, n_init, features_dir, path_to_csv)

def do_simulation_svm (centers_train, centers_test, k, max_iter, n_init, features_dir, path_to_csv):
    #load the features of train data
    coords_train, features_train, n_points_vect_train, patient_id_vect_train = file_manager.load_data(centers_train, features_dir)

    #apply k-means
    kmeans_model, clusters, centroids = k_means_batch.k_means(k, max_iter, n_init, features_train)
    
    #Histograms normalized for every image
    X_train = k_means_batch.make_histo_image_level(clusters, k, n_points_vect_train)
    
    y_train = file_manager.load_binary_labels(path_to_csv, centers_train, exclude_pairs=[(35,2)])
    
    print(y_train)

    #save histograms
    #file_manager.save_hist(X_train, hist_dir, centers_train)

    
    #load the features of test data
    coords_test, features_test, n_points_vect_test, patient_id_vect_test = file_manager.load_data(centers_test, features_dir)

    #Using centroids
    clusters_test = k_means_batch.cluster_from_centroids(kmeans_model, features_test)
    X_test = k_means_batch.make_histo_image_level(clusters_test, k, n_points_vect_test)
    y_test = file_manager.load_binary_labels(path_to_csv, centers_test, exclude_pairs=[(35,2)])

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    #learn
    accuracy, classification_report, confusion_matrix = predictor.do_SVM(X_train, X_test, y_train, y_test)

    print(f"Accuracy : {accuracy}\n, Classification report : \n{classification_report}\n, Confusion matrix : \n{confusion_matrix}\n")
    #TODO
    # file_manager.save_metrics(accuracy, classification_report, confusion_matrix, excel_metrics)

def do_simulation_tree(centers_train, centers_test, k, max_iter, n_init, features_dir, path_to_csv):
    
    #load the features of train data
    all_centers = centers_train+centers_test
    coords_train, features_train, n_points_vect_train, patient_id_vect_train = file_manager.load_data(all_centers, features_dir)

    #apply k-means
    kmeans_model, clusters, centroids = k_means_batch.k_means(k, max_iter, n_init, features_train)
    
    #Histograms normalized for every image
    X = k_means_batch.make_histo_image_level(clusters, k, n_points_vect_train)
    
    y= file_manager.load_multi_labels(path_to_csv, centers_train, exclude_pairs=[(35,2)])
    
    print(np.unique(y))
    #TODO : show X
    features_names = ['C0','C1','C2','C3','C4']
    predictor.draw_decision_tree(X,y, feature_names= features_names, class_names=np.unique(y), max_depth=6, min_samples_split=6, min_samples_leaf=6, max_leaf_nodes=None)
    predictor.draw_decision_tree(X,y, feature_names= features_names, class_names=np.unique(y), max_depth=6, min_samples_split=10, min_samples_leaf=6, max_leaf_nodes=None)
    predictor.draw_decision_tree(X,y, feature_names= features_names, class_names=np.unique(y), max_depth=6, min_samples_split=4, min_samples_leaf=6, max_leaf_nodes=None)
    predictor.draw_decision_tree(X,y, feature_names= features_names, class_names=np.unique(y), max_depth=6, min_samples_split=2, min_samples_leaf=6, max_leaf_nodes=None)
    
def do_nn(centers_train, centers_test, k, max_iter, n_init, features_dir, path_to_csv, binary=False):
    #load the features of train data
    coords_train, features_train, n_points_vect_train, patient_id_vect_train = file_manager.load_data(centers_train, features_dir)

    #apply k-means
    kmeans_model, clusters, centroids = k_means_batch.k_means(k, max_iter, n_init, features_train)
    
    #Histograms normalized for every image
    X_train = k_means_batch.make_histo_image_level(clusters, k, n_points_vect_train)
    
    if binary :
        y_train = file_manager.load_binary_labels(path_to_csv, centers_train, exclude_pairs=[(35,2)])
    else:
        y_train = file_manager.load_multi_labels(path_to_csv, centers_train, exclude_pairs=[(35,2)])
    
    print(y_train)
    
    #load the features of test data
    coords_test, features_test, n_points_vect_test, patient_id_vect_test = file_manager.load_data(centers_test, features_dir)

    #Using centroids
    clusters_test = k_means_batch.cluster_from_centroids(kmeans_model, features_test)
    X_test = k_means_batch.make_histo_image_level(clusters_test, k, n_points_vect_test)
    if binary :
        y_test = file_manager.load_binary_labels(path_to_csv, centers_test, exclude_pairs=[(35,2)])
    else :
        y_test = file_manager.load_multi_labels(path_to_csv, centers_test, exclude_pairs=[(35,2)])

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    network.use_nn(X_train, y_train, X_test, y_test)
    
def main():

    #Select parameters of clustering
    k, max_iter, centers_train, centers_test, n_init  = param_choice()
    
    #Select computer
    computer = platform.node().upper()
    features_dir, path_to_csv, hist_dir, excel_metrics = file_manager.assign_paths(computer)

    #SVM
    #circular_train_test(k, max_iter, n_init, features_dir, path_to_csv)

    #Decision tree :
    #do_simulation_tree(centers_train, centers_test, k, max_iter, n_init, features_dir, path_to_csv)

    #NN
    do_nn(centers_train, centers_test, k, max_iter, n_init, features_dir, path_to_csv, binary = True)
    return 0

if __name__ == '__main__':
    main()