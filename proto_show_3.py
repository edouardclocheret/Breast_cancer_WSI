import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import h5py
import os
from tqdm import tqdm
import tifffile as tiff

import k_means_3


def main ():
    
    #k_means
    k=5
    max_iter =2
    n_sane =2
    n_unsane =2
    centers =[0,1]
    coords, clusters, n_points_vect, patient_id_vect = k_means_3.k_means(k,max_iter, centers, n_sane, n_unsane)
    
    #Prototype extraction
    size =256
    patient =0
    node =0
    #ki = 0
    for ki in range(5):
        print(f"Exctracting prototype of {ki}")
        proto = prototype(ki,coords, clusters, size, patient, node)

        #Display
        title = "proto_"+str(ki)+"_patient_"+str(patient).zfill(3)+"_node_"+str(node)
        plt.figure(title)
        plt.imshow(proto)

        
        label = "Prototype of cluster "+str(ki)
        plt.xlabel(label)
        plt.show()

    return 0


def prototype(ki,coords, clusters, size, patient, node):


    #this low resolution image is used to check if it is not background
    name = "C:\Edouard\Tecnico\Code\Train_features\stitches"+"\patient_"+str(patient).zfill(3)+"_node_"+str(node)+".jpg"
    image = plt.imread(name)

    ntile = 0
    x1 = coords[ntile,0]//64
    y1 = coords[ntile,1]//64
    x2 = x1 +256//64
    y2 = y1 +256//64
    selection = image[y1:y2, x1:x2, :]

    # Searching for a tile in cluster ki
    while clusters[ntile] != ki or selection.sum() < 4000:
        ntile+=1      
        x1 = coords[ntile,0]//64
        y1 = coords[ntile,1]//64
        x2 = x1 +256//64
        y2 = y1 +256//64
        selection = image[y1:y2, x1:x2, :]
        
    
    #This is the full-resolution image
    image_path = "C:\Edouard\TECNICO\Code\Train_features_full\patient_"+str(patient).zfill(3)+"\patient_"+str(patient).zfill(3)+"_node_"+str(node)+".tif"
    
    # Define the region to read (top, left, height, width)
    region = (coords[ntile,1], coords[ntile,0], size, size)
    
    return read_tiff_region(image_path, region)
    

# Function to read a region of a TIFF image
def read_tiff_region(image_path, region):
    with tiff.TiffFile(image_path) as tif:
        # Access the first page of the TIFF file
        page = tif.pages[0]
        # Get the image dimensions
        image_shape = page.shape
        # Ensure the region is within the image bounds
        top, left, height, width = region
        bottom = top + height
        right = left + width
        if bottom > image_shape[0] or right > image_shape[1]:
            raise ValueError("Region is out of image bounds")
        # Read the region
        region_data = page.asarray()[top:bottom, left:right]
        return region_data


if __name__ == '__main__':
    main()
