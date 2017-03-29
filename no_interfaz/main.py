# Librerias
import cv2
import numpy as np
from random import randint
import math
from PIL import Image, ImageDraw


class Montecarlo:

    centrosGeometricos = []
    esquinasFiguras = []

    def abrirImagen(self, ruta):
        self.archivo = ruta
        self.imagen = cv2.imread(self.archivo)

    def buscarContornos(self):
        hsv = cv2.cvtColor(self.imagen, cv2.COLOR_BGR2HSV)
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

        # Difuminamos la mascara para suavizar los contornos y aplicamos filtro canny
        blur = cv2.GaussianBlur(mask, (5, 5), 0)
        edges = cv2.Canny(blur, 1, 2)

        # Extracción de contornos por figura
        _, contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return contours, mask

    def calcularAreas(self, contours, mask):
        areas = [cv2.contourArea(c) for c in contours]
        i = 0
        for extension in areas:
            actual = contours[i]
            approx = cv2.approxPolyDP(actual, 0.05 * cv2.arcLength(actual, True), True)
            i = i + 1
            print('-----------------Coordenadas----------------')
            print('Figura ' + str(i) + ': ')
            print(str(approx).replace('[[', '(').replace(']]', ')').replace('([', ' (').replace(')]', ')'))
            print()

    def calcularCentros(self, contours, experiments):
        for actual in contours:
            if cv2.contourArea(actual) > 10:
                approx = cv2.approxPolyDP(actual, 0.05 * cv2.arcLength(actual, True), True)
                xmax = 0
                ymax = 0
                xmin, ymin, channels = self.imagen.shape
                for points in approx:
                    if xmax < points[:, 0]:
                        xmax = points[:, 0]
                    if xmin > points[:, 0]:
                        xmin = points[:, 0]
                    if ymax < points[:, 1]:
                        ymax = points[:, 1]
                    if ymin > points[:, 1]:
                        ymin = points[:, 1]
                self.esquinasFiguras.append([xmin, ymin, xmax, ymax])
                if xmax - xmin > ymax - ymin:
                    differencemax = (xmax - xmin / 2) + 10
                else:
                    differencemax = (ymax - ymin / 2) + 10
                randomDictionary = {}
                for i in range(experiments):
                    distancelist = []

                    # Se genera un punto aleatorio (x,y)
                    xa = randint(xmin, xmax)
                    ya = randint(ymin, ymax)

                    # Se calculan las distancias entre el punto aleatorio y los puntos de referencia
                    for p in approx:
                        distance = math.sqrt(pow(p[:, 0] - xa, 2) + pow(p[:, 1] - ya, 2))
                        distancelist.append(distance)
                    # Si la diferencia de distancias es menor a la establecida, se guarda el punto
                    if (max(distancelist) - min(distancelist)) <= differencemax:
                        randomDictionary[xa] = [ya]
                pointslist = randomDictionary.items()

                # Se calcula el promedio de los puntos validos
                summationx = 0
                summationy = 0
                for x, y in pointslist:
                    summationx += x
                    summationy += y[0]
                averagex = summationx / len(pointslist)
                averagey = summationy / len(pointslist)

                print('Centro geometrico -> x: ' + str(averagex) + 'y: ' + str(averagey))
                self.centrosGeometricos.append([averagex, averagey])

    def procesar(self, rutaImagen, experiments):
        self.abrirImagen(rutaImagen)
        contours, mask = self.buscarContornos()
        self.calcularAreas(contours, mask)
        self.calcularCentros(contours, experiments)

        cv2.namedWindow("Centros geometricos")
        cv2.setMouseCallback("Centros geometricos", self.mouse)
        # Salir con ESC
        while(1):
            # Mostrar la mascara final y la imagen
            # cv2.imshow('Figuras detectadas', mask)
            cv2.imshow('Centros geometricos', self.imagen)
            tecla = cv2.waitKey(5) & 0xFF
            if tecla == 27:
                break

        cv2.destroyAllWindows()

    def mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # inicializamos el dibujo
            imagenres = Image.open(self.archivo)
            dibujo = ImageDraw.Draw(imagenres)

            i = 0
            for esquina in self.esquinasFiguras:
                if x >= esquina[0] and x <= esquina[2] and y >= esquina[1] and y <= esquina[3]:  # El clic esta dentro del perimetro
                    centro = self.centrosGeometricos[i]
                    dibujo.text((centro[0], centro[1]), 'x', fill="black")
                    pass
                pass
                i += 1

            # Guardamos el dibujo
            imagenres.save("linea.png")
            self.imagen = cv2.imread('linea.png')


montecarlo = Montecarlo()
montecarlo.procesar('irregulares.jpg', 1000)
# montecarlo.procesar('colores.png', 1000)
# montecarlo.procesar('cuadrado.png', 1000)
# montecarlo.procesar('figuras.png', 1000)
# montecarlo.procesar('mapa5.png', 1000)
# montecarlo.procesar('rompecabezas3.jpg', 1000)
