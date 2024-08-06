import matplotlib.pyplot as plt


image = plt.imread("C:\Edouard\Tecnico\Code\Praca_comercio.jpg")

x1 = 60
y1 = 60
x2 = x1 +200
y2 = y1 +200
print(x1,x2)
print(y1,y2)
selection = image[y1:y2, x1:x2, :]

print(selection)
title = "Selection"
print(title)
plt.figure(title)
plt.imshow(selection)
plt.axis('off')
plt.axis('image')
plt.show()

plt.figure()
plt.imshow(image)
plt.scatter([y1,y2],[x1,x2],s=20)
plt.show()