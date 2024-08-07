import matplotlib.pyplot as plt
import tifffile as tiff
import numpy as np

def prototype(ki,coords, clusters, size, patient, node):


    #this low resolution image is used to check if it is not background
    name = "C:\Edouard\Tecnico\Code\Train_features\stitches"+"\patient_"+str(patient).zfill(3)+"_node_"+str(node)+".jpg"
    image = plt.imread(name)

    #can be artificially initialized to change region of search
    ntile = 670
    
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
        
    print(f"ntile = {ntile}")
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

coords = loaded_arr = np.load('coords.npy')
clusters = np.load('clusters.npy')

#Prototype extraction
size =256
patient =0
node =0
k=5

for ki in [4]:
    print(f"Exctracting prototype of {ki}")
    proto = prototype(ki,coords, clusters, size, patient, node)

    plt.imshow(proto)
    plt.show()
    del proto #Mandatory to release memory, to avoid crashes