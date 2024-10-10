# import the necessary packages
from nms import non_max_suppression_slow
import numpy as np
import cv2


# construct a list containing the images that will be examined
# along with their respective bounding boxes
# Lista de imágenes con sus cuadros delimitadores
images = [
    (
        "images/img1.jpg",
        np.array(
            [
                (12, 84, 140, 212),
                (24, 84, 152, 212),
                (36, 84, 164, 212),
                (12, 96, 140, 224),
                (24, 96, 152, 224),
                (24, 108, 152, 236),
            ]
        ),
    ),
    (
        "images/img2.jpg",
        np.array([(114, 60, 178, 124), (120, 60, 184, 124), (114, 66, 178, 130)]),
    ),
    (
        "images/img3.jpg",
        np.array(
            [
                (12, 30, 76, 94),
                (12, 36, 76, 100),
                (72, 36, 200, 164),
                (84, 48, 212, 176),
            ]
        ),
    ),
]
# Bucle sobre las imágenes
for imagePath, boundingBoxes in images:
    # Cargar la imagen y clonarla
    print("[x] %d initial bounding boxes" % (len(boundingBoxes)))
    image = cv2.imread(imagePath)
    if image is None:
        print(f"Error loading image: {imagePath}")
        continue
    orig = image.copy()

    # Dibujar los cuadros delimitadores originales en la imagen clonada
    for startX, startY, endX, endY in boundingBoxes:
        cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 0, 255), 2)

    # Aplicar Non-Maximum Suppression (NMS)
    pick = non_max_suppression_slow(boundingBoxes, 0.3)
    print("[x] after applying non-maximum, %d bounding boxes" % (len(pick)))

    # Dibujar los cuadros seleccionados después de NMS
    for startX, startY, endX, endY in pick:
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # Mostrar las imágenes originales y después de NMS
    cv2.imshow("Original", orig)
    cv2.imshow("After NMS", image)
    cv2.waitKey(0)
