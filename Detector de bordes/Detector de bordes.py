#Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2

imagen = cv2.imread('images/kratos.jpg',0)
cv2.imshow('Imagen original', imagen)
cv2.waitKey(0)
