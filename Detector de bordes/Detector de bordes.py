# Import the libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Cargamos la imagen
image = cv2.imread('images/kratos1.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print(np.shape(image))
img1 = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
plt.figure(1)
plt.title("Original")
plt.imshow(image)
plt.figure(2)
plt.title("Blanco y negro")
plt.imshow(img1, cmap='gray')
# Definimos los filtros
# Vertical
w1 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
# Horizontal
w2 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
# Diagonal
w3 = np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]])
forma = np.shape(img1)
print(forma[0])
# Inicializamos la imagenes resultantes como matrices de ceros
vertical = np.zeros((img1.shape[0], img1.shape[1]))
horizontal = np.zeros((img1.shape[0], img1.shape[1]))
Diagonal = np.zeros((img1.shape[0], img1.shape[1]))
# Ciclo para recorrer la imagen
for i in range(1, img1.shape[0] - 1):
    for j in range(1, img1.shape[1] - 1):
        # Convolucion con el sobel vertical
        mv = (img1.item(i - 1, j - 1) * w1[0][0] + img1.item(i - 1, j) * w1[0][1] + img1.item(i - 1, j + 1) * w1[0][2]
              + img1.item(i, j - 1) * w1[1][0] + img1.item(i, j) * w1[1][1] + img1.item(i, j + 1) * w1[1][2]
              + img1.item(i + 1, j - 1) * w1[2][0] + img1.item(i + 1, j) * w1[2][1] + img1.item(i + 1, j + 1) * w1[2][
                  2])
        # Comprobar si los valores estan entre 0 y 255
        if mv > 255:
            mv = 255
        elif mv < 0:
            mv = 0
        # Convolucion con el sobel horizontal
        mh = (img1.item(i - 1, j - 1) * w2[0][0] + img1.item(i - 1, j) * w2[0][1] + img1.item(i - 1, j + 1) * w2[0][2]
              + img1.item(i, j - 1) * w2[1][0] + img1.item(i, j) * w2[1][1] + img1.item(i, j + 1) * w2[1][2]
              + img1.item(i + 1, j - 1) * w2[2][0] + img1.item(i + 1, j) * w2[2][1] + img1.item(i + 1, j + 1) * w2[2][
                  2])
        # Comprobar si los valores estan entre 0 y 255
        if mh > 255:
            mh = 255
        elif mh < 0:
            mh = 0
        # Convolucion con el sobel Diagonal
        md = (img1.item(i - 1, j - 1) * w3[0][0] + img1.item(i - 1, j) * w3[0][1] + img1.item(i - 1, j + 1) * w3[0][2]
              + img1.item(i, j - 1) * w3[1][0] + img1.item(i, j) * w3[1][1] + img1.item(i, j + 1) * w3[1][2]
              + img1.item(i + 1, j - 1) * w3[2][0] + img1.item(i + 1, j) * w3[2][1] + img1.item(i + 1, j + 1) * w3[2][
                  2])
        # Comprobar si los valores estan entre 0 y 255
        if md > 255:
            md = 255
        elif md < 0:
            md = 0
        # Asignacion de los valores a los pixeles de la imagen resultante
        vertical.itemset((i, j), mv)
        horizontal.itemset((i, j), mh)
        Diagonal.itemset((i, j), md)

plt.figure(3)
plt.title("bordes sobel verticales")
plt.imshow(vertical, cmap='gray')
plt.figure(4)
plt.title("bordes sobel hotizontal")
plt.imshow(horizontal, cmap='gray')
plt.figure(5)
plt.title("bordes sobel Diagonal")
plt.imshow(Diagonal, cmap='gray')
# display that image
plt.show()
