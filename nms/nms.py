# Importar los paquetes necesarios
import numpy as np


def non_max_suppression_slow(boxes, overlapThresh):
    # boxes: matriz de cuadros delimitadores con las coordenadas [x1, y1, x2, y2]
    # overlapThresh: umbral de superposición para decidir cuándo suprimir un cuadro

    # Si no hay cuadros, devuelve una lista vacía.
    if len(boxes) == 0:
        return []

    # Inicializa la lista de índices seleccionados (pick).
    pick = []

    # Toma las coordenadas (x1, y1, x2, y2) de los cuadros delimitadores.
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # Calcula el área de los cuadros delimitadores.
    # (ancho = x2 - x1 + 1) * (alto = y2 - y1 + 1)
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    # Ordena los índices por la coordenada y2 (parte inferior derecha del cuadro).
    idxs = np.argsort(y2)

    # Ejemplo:
    # Supongamos que tenemos 3 cuadros delimitadores:
    # boxes = [[10, 20, 40, 60], [15, 25, 42, 58], [50, 50, 90, 90]]
    # Esto se traduce en:
    # x1 = [10, 15, 50], y1 = [20, 25, 50], x2 = [40, 42, 90], y2 = [60, 58, 90]
    # El área sería:
    # area = [(40-10+1)*(60-20+1), (42-15+1)*(58-25+1), (90-50+1)*(90-50+1)]
    # area = [1241, 1014, 1681]

    # Iterar mientras queden índices en la lista de cuadros delimitadores.
    while len(idxs) > 0:
        # Toma el último índice (el cuadro con el mayor y2).
        last = len(idxs) - 1
        i = idxs[last]

        # Agrega este índice a la lista de seleccionados (pick).
        pick.append(i)

        # Lista de índices que vamos a suprimir.
        suppress = [last]

        # Recorre todos los otros cuadros para comprobar si deben ser suprimidos.
        for pos in range(0, last):
            # Toma el índice actual.
            j = idxs[pos]

            # Encuentra las coordenadas máximas y mínimas entre los cuadros `i` y `j`.
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])

            # Calcula el ancho y la altura del cuadro de superposición.
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)

            # Calcula la relación de superposición (Intersection over Union, IoU).
            overlap = float(w * h) / area[j]

            # Si la superposición es mayor que el umbral dado, añade a la lista de supresión.
            if overlap > overlapThresh:
                suppress.append(pos)

        # Elimina los índices suprimidos de la lista de índices.
        idxs = np.delete(idxs, suppress)

    # Devuelve solo los cuadros que no fueron suprimidos.
    return boxes[pick]


# Ejemplo de uso:
# boxes = np.array([[10, 20, 40, 60], [15, 25, 42, 58], [50, 50, 90, 90]])
# overlapThresh = 0.3

# Aplicamos NMS para eliminar cuadros superpuestos.
# filtered_boxes = non_max_suppression_slow(boxes, overlapThresh)
# print(filtered_boxes)

""" El algoritmo ordena estos cuadros según la coordenada y2 y luego evalúa cada par para ver si su superposición excede el umbral (overlapThresh). Si dos cuadros tienen suficiente superposición, se elimina el menos importante (usualmente el que tiene el área más pequeña o está más abajo).

En este caso, los dos primeros cuadros (que se superponen parcialmente) se evalúan, y uno de ellos puede ser eliminado según el umbral de superposición.

Al final, el resultado será una lista de cuadros que no se solapan por encima del umbral indicado, devolviendo solo los más importantes."""
