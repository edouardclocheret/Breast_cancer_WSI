import h5py
import cv2
import matplotlib.pyplot as plt

"""f = h5py.File("C:\Edouard\Tecnico\Code\patient_000_node_0.h5", 'r')

print(f.keys())
coords = f['coords']
print(coords)

print(max(coords[0]))"""

name = "C:\Edouard\Tecnico\Code\patient-0-files-features-masks\patient_000_node_0.jpg"
image = cv2.imread(name)
print(image.shape)
plt.figure()
plt.imshow(image)
plt.show()

image_2 = image[  2000:2040, 1500:1540, :]
plt.figure()
plt.imshow(image_2)
plt.show()