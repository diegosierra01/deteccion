import cv2
import numpy as np
from matplotlib import pyplot as plt



img = cv2.imread('imagen2.jpg')

img = cv2.resize(img,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)

#cv2.imshow("sisas", img)


resultado = cv2.pyrMeanShiftFiltering(img, 20, 40, 4)
cv2.imshow("primera", resultado)

cantidadFiltros = 2
for x in xrange(0,cantidadFiltros):
	resultado = cv2.pyrMeanShiftFiltering(resultado, 40, 80, 4)
	pass

#cv2.imshow("segunda", resultado)

gray_image = cv2.cvtColor(resultado, cv2.COLOR_BGR2GRAY)
cv2.imshow("Ultima iteracion", gray_image)

plt.subplot(311)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.hist(img.ravel(),256,[0,256])


plt.subplot(312)
plt.hist(img.ravel(),256,[0,256])

plt.subplot(313)
plt.hist(gray_image.ravel(),256,[0,256])
plt.title("Histograma ultima iteracion")
plt.show()