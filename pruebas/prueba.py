import cv2
import numpy as np
from matplotlib import pyplot as plt
import time


def meanShift():

	#img = cv2.imread('deteccion/figuras.png')
	img = cv2.imread('deteccion/mapa4.png')

	img = cv2.resize(img,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)

	#cv2.imshow("sisas", img)


	resultado = cv2.pyrMeanShiftFiltering(img, 25, 40, 3)
	#cv2.imshow("primera", resultado)

	cantidadFiltros = 1
	for x in xrange(0,cantidadFiltros):
		resultado = cv2.pyrMeanShiftFiltering(resultado, 25, 35, 3)
		pass

	#cv2.imshow("segunda", resultado)

	imgray = cv2.cvtColor(resultado,cv2.COLOR_BGR2GRAY)
	bordes = cv2.Canny(imgray, 30, 200)
	
	ret,thresh = cv2.threshold(imgray,127,255,0)
	
	im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

	cv2.drawContours(resultado, contours, -1, (0,255,0), 3)


	hsv = cv2.cvtColor(resultado, cv2.COLOR_BGR2HSV)

	# Rango de colores detectados:
	# Verdes:
	verde_bajos = np.array([49, 50, 50], dtype=np.uint8)
	verde_altos = np.array([107, 255, 255], dtype=np.uint8)
	# Azules:
	azul_bajos = np.array([100, 65, 75], dtype=np.uint8)
	azul_altos = np.array([180, 255, 255], dtype=np.uint8)
	# Rojos:
	rojo_bajos1 = np.array([0, 65, 75], dtype=np.uint8)
	rojo_altos1 = np.array([12, 255, 255], dtype=np.uint8)
	rojo_bajos2 = np.array([240, 65, 75], dtype=np.uint8)
	rojo_altos2 = np.array([256, 255, 255], dtype=np.uint8)
	# Amarillos:
	amarillo_bajos = np.array([16, 76, 72], dtype=np.uint8)
	amarillo_altos = np.array([30, 255, 255], dtype=np.uint8)
	# Morados:
	morado_bajos = np.array([100, 65, 30], dtype=np.uint8)
	morado_altos = np.array([150, 255, 255], dtype=np.uint8)
	# Naranjas:
	naranja_bajos = np.array([5, 50, 50], dtype=np.uint8)
	naranja_altos = np.array([15, 255, 255], dtype=np.uint8)

	# Crear las mascaras
	mascara_verde = cv2.inRange(hsv, verde_bajos, verde_altos)
	mascara_rojo1 = cv2.inRange(hsv, rojo_bajos1, rojo_altos1)
	mascara_rojo2 = cv2.inRange(hsv, rojo_bajos2, rojo_altos2)
	mascara_azul = cv2.inRange(hsv, azul_bajos, azul_altos)
	mascara_amarillo = cv2.inRange(hsv, amarillo_bajos, amarillo_altos)
	mascara_morado = cv2.inRange(hsv, morado_bajos, morado_altos)
	mascara_naranja = cv2.inRange(hsv, naranja_bajos, naranja_altos)

	# Juntar todas las mascaras
	mask = cv2.add(mascara_rojo1, mascara_rojo2)
	mask = cv2.add(mask, mascara_amarillo)
	mask = cv2.add(mask, mascara_verde)
	mask = cv2.add(mask, mascara_azul)
	mask = cv2.add(mask, mascara_morado)
	mask = cv2.add(mask, mascara_naranja)

	kernel = np.ones((15,15),np.uint8)
	mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
	mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
	#mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

	# Difuminamos la mascara para suavizar los contornos y aplicamos filtro canny
	blur = cv2.GaussianBlur(mask, (5, 5), 0)
	edges = cv2.Canny(mask, 1, 2)

	# Si el area blanca de la mascara es superior a 500px, no se trata de ruido
	#contours, hier = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	sfs, contours, hier = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cv2.drawContours(resultado, contours, -1, (0,255,0), 3)

	#cv2.imshow("Bordes", bordes)
	#cv2.imshow("Ultima", imgray)
	#cv2.imshow("Ultima", resultado)
	cv2.imshow("Ultima", resultado)

	plt.subplot(411)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	plt.hist(img.ravel(),256,[0,256])


	plt.subplot(412)
	plt.hist(img.ravel(),256,[0,256])

	plt.subplot(413)
	#plt.hist(gray_image.ravel(),256,[0,256])
	plt.hist(resultado.ravel(),256,[0,256])

	plt.subplot(414)
	plt.hist(imgray.ravel(),256,[0,256])
	print("--- %s seconds ---" % (time.time() - start_time))

	plt.show()


start_time = time.time()
meanShift()

#nuevo()