# Librerias
import cv2
import numpy as np

# Caputrar una imagen y convertirla a hsv
# imagen = cv2.imread('figuras.png')
imagen = cv2.imread('colores.png')
hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

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
edges = cv2.Canny(mask, 1, 2)

# Si el area blanca de la mascara es superior a 500px, no se trata de ruido
_, contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
areas = [cv2.contourArea(c) for c in contours]
i = 0
for extension in areas:
    actual = contours[i]
    approx = cv2.approxPolyDP(actual, 0.05 * cv2.arcLength(actual, True), True)
    cv2.drawContours(imagen, [actual], 0, (0, 0, 255), 2)
    cv2.drawContours(mask, [actual], 0, (0, 0, 255), 2)
    i = i + 1


# Salir con ESC
while(1):
    # Mostrar la mascara final y la imagen
    cv2.imshow('Finale', mask)
    cv2.imshow('Imagen', imagen)
    tecla = cv2.waitKey(5) & 0xFF
    if tecla == 27:
        break

cv2.destroyAllWindows()
