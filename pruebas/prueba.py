import cv2
import numpy as np
from matplotlib import pyplot as plt
import time


def meanShift():

	img = cv2.imread('imagen2.jpg')

	img = cv2.resize(img,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)

	#cv2.imshow("sisas", img)


	resultado = cv2.pyrMeanShiftFiltering(img, 30, 30, 3)
	cv2.imshow("primera", resultado)

	cantidadFiltros = 5
	for x in xrange(0,cantidadFiltros):
		resultado = cv2.pyrMeanShiftFiltering(resultado, 30, 30, 3)
		pass

	#cv2.imshow("segunda", resultado)

	gray_image = cv2.cvtColor(resultado, cv2.COLOR_BGR2GRAY)
	cv2.imshow("Ultima", gray_image)
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
	plt.hist(gray_image.ravel(),256,[0,256])
	print("--- %s seconds ---" % (time.time() - start_time))

	plt.show()


start_time = time.time()
meanShift()

#nuevo()