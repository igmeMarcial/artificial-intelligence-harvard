import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter


def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Calcula el Intersection over Union (IoU) entre las cajas predichas y las cajas reales.

    Parámetros:
        boxes_preds (tensor): Predicciones de cajas delimitadoras (BATCH_SIZE, 4), cada fila contiene [x, y, ancho, alto].
        boxes_labels (tensor): Etiquetas reales de las cajas delimitadoras (BATCH_SIZE, 4).
        box_format (str): Formato de las cajas, puede ser 'midpoint' (centro y tamaño) o 'corners' (esquinas).

    Retorna:
        tensor: IoU entre las cajas predichas y las reales.
    """

    # Si las cajas están en formato 'midpoint' (centro, ancho y alto), convertimos al formato (x1, y1, x2, y2)
    if box_format == "midpoint":
        box1_x1 = (
            boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        )  # x1 = x_center - width / 2
        box1_y1 = (
            boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        )  # y1 = y_center - height / 2
        box1_x2 = (
            boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        )  # x2 = x_center + width / 2
        box1_y2 = (
            boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        )  # y2 = y_center + height / 2
        box2_x1 = (
            boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        )  # Lo mismo para las etiquetas
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    # Si las cajas están en formato 'corners' (esquinas), tomamos las coordenadas tal como están
    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    # Calcula las coordenadas del área de intersección
    x1 = torch.max(
        box1_x1, box2_x1
    )  # Máxima coordenada de la esquina superior izquierda
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)  # Mínima coordenada de la esquina inferior derecha
    y2 = torch.min(box1_y2, box2_y2)

    # Calcula el área de intersección (si las cajas no se superponen, el área será 0)
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    # Calcula el área de cada caja (cajas 1 y 2)
    box1_area = abs(
        (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    )  # Área de la caja predicha
    box2_area = abs(
        (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    )  # Área de la caja etiquetada

    # Retorna el IoU: área de intersección dividido entre la suma de las áreas menos la intersección
    return intersection / (
        box1_area + box2_area - intersection + 1e-6
    )  # 1e-6 es para evitar división por cero


def non_max_suppression(bboxes, iou_threshold, threshold, box_format="corners"):
    """
    Realiza Non-Maximum Suppression (NMS) para eliminar las cajas delimitadoras redundantes.

    Parámetros:
        bboxes (list): Lista de listas que contienen las cajas en formato [clase_pred, probabilidad, x1, y1, x2, y2].
        iou_threshold (float): Umbral de IoU para determinar si una caja es redundante.
        threshold (float): Umbral de probabilidad para filtrar predicciones con baja probabilidad.
        box_format (str): "midpoint" o "corners" para especificar el formato de las cajas.

    Retorna:
        list: Lista de las cajas después de aplicar Non-Maximum Suppression.
    """

    assert type(bboxes) == list  # Verifica que las cajas estén en una lista

    # Filtra las cajas cuya probabilidad es mayor al umbral
    bboxes = [box for box in bboxes if box[1] > threshold]

    # Ordena las cajas por la probabilidad de predicción en orden descendente
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)

    bboxes_after_nms = []  # Lista para almacenar las cajas seleccionadas

    # Mientras haya cajas en la lista
    while bboxes:
        chosen_box = bboxes.pop(0)  # Elige la caja con mayor probabilidad

        # Filtra las cajas que no pertenecen a la misma clase o que tienen IoU bajo
        bboxes = [
            box
            for box in bboxes
            if box[0]
            != chosen_box[0]  # Si la clase es diferente, no es necesario eliminarla
            or intersection_over_union(
                torch.tensor(
                    chosen_box[2:]
                ),  # Compara la caja elegida con el resto usando IoU
                torch.tensor(box[2:]),
                box_format=box_format,
            )
            < iou_threshold  # Elimina las cajas con IoU mayor al umbral
        ]

        bboxes_after_nms.append(chosen_box)  # Agrega la caja elegida a la lista final

    return bboxes_after_nms  # Retorna las cajas seleccionadas después de aplicar NMS


def mean_average_precision(
    pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=20
):
    """
    Calcula el mean Average Precision (mAP), que es una métrica usada en detección de objetos
    para evaluar la precisión media en todas las clases y umbrales de IoU.

    Parámetros:
        pred_boxes (list): Lista de listas que contiene todas las predicciones de las cajas delimitadoras.
                           Cada predicción está en el formato [train_idx, clase_pred, prob_score, x1, y1, x2, y2].
        true_boxes (list): Similar a pred_boxes, pero para las cajas reales (ground truth).
        iou_threshold (float): Umbral de IoU para determinar si una predicción es correcta.
        box_format (str): Especifica el formato de las cajas ("midpoint" o "corners").
        num_classes (int): Número de clases en el dataset.

    Retorna:
        float: El valor del mAP para todas las clases dado un umbral de IoU específico.
    """

    # Lista para almacenar la precisión promedio para cada clase.
    average_precisions = []
    epsilon = 1e-6  # Para evitar divisiones por cero.

    for c in range(num_classes):
        detections = []  # Lista de predicciones para la clase actual.
        ground_truths = []  # Lista de cajas reales para la clase actual.

        # Recopila todas las predicciones y cajas reales de la clase actual.
        for detection in pred_boxes:
            if (
                detection[1] == c
            ):  # Si la clase de la predicción es igual a la clase actual.
                detections.append(detection)

        for true_box in true_boxes:
            if (
                true_box[1] == c
            ):  # Si la clase de la caja real es igual a la clase actual.
                ground_truths.append(true_box)

        # Cuenta cuántas cajas reales hay por cada imagen de entrenamiento.
        # amount_bboxes es un diccionario con la cantidad de cajas reales por imagen.
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # Inicializa un tensor de ceros para cada caja real de cada imagen (no detectada inicialmente).
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # Ordena las predicciones según la probabilidad de la caja, de mayor a menor.
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))  # Tensor de verdaderos positivos.
        FP = torch.zeros((len(detections)))  # Tensor de falsos positivos.
        total_true_bboxes = len(ground_truths)  # Número total de cajas reales.

        # Si no hay cajas reales para esta clase, pasa a la siguiente iteración.
        if total_true_bboxes == 0:
            continue

        # Para cada predicción de caja...
        for detection_idx, detection in enumerate(detections):
            # Obtén las cajas reales de la misma imagen de entrenamiento.
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)  # Número de cajas reales en esa imagen.
            best_iou = 0  # Mejor IoU encontrado.

            # Compara la predicción con todas las cajas reales de la imagen para encontrar el mejor IoU.
            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),  # Predicción.
                    torch.tensor(gt[3:]),  # Caja real.
                    box_format=box_format,
                )

                # Si este IoU es mejor, lo actualiza.
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            # Si el mejor IoU supera el umbral y la caja real no ha sido detectada antes...
            if best_iou > iou_threshold:
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 1  # Verdadero positivo.
                    amount_bboxes[detection[0]][
                        best_gt_idx
                    ] = 1  # Marca la caja real como detectada.
                else:
                    FP[detection_idx] = (
                        1  # Falso positivo (ya detectada anteriormente).
                    )
            else:
                FP[detection_idx] = 1  # Falso positivo (IoU menor al umbral).

        # Calcula las sumas acumulativas de verdaderos y falsos positivos.
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)

        # Calcula el recall y la precisión acumulada.
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))

        # Asegura que la precisión y el recall inicien en 1 y 0 respectivamente.
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))

        # Calcula la precisión promedio usando integración numérica (trapezoidal).
        average_precisions.append(torch.trapz(precisions, recalls))

    # Retorna la media de todas las precisiones promedio.
    return sum(average_precisions) / len(average_precisions)


def plot_image(image, boxes):
    """
    Dibuja las cajas delimitadoras predichas sobre la imagen.

    Parámetros:
        image (numpy array): La imagen sobre la cual se dibujarán las cajas.
        boxes (list): Lista de cajas en formato [clase_pred, prob_score, x, y, ancho, alto].
    """

    im = np.array(image)
    height, width, _ = im.shape  # Dimensiones de la imagen.

    # Crear la figura y el eje.
    fig, ax = plt.subplots(1)
    ax.imshow(im)  # Mostrar la imagen.

    # Para cada caja en boxes...
    for box in boxes:
        box = box[2:]  # Extraer las coordenadas de la caja.
        assert len(box) == 4, "La caja debe tener formato [x, y, ancho, alto]"

        # Convertir las coordenadas del formato [x, y, ancho, alto] al formato de rectángulo.
        upper_left_x = box[0] - box[2] / 2  # Coordenada superior izquierda x.
        upper_left_y = box[1] - box[3] / 2  # Coordenada superior izquierda y.

        # Crear un parche rectangular para dibujar la caja.
        rect = patches.Rectangle(
            (
                upper_left_x * width,
                upper_left_y * height,
            ),  # Escalar según el tamaño de la imagen.
            box[2] * width,  # Ancho de la caja.
            box[3] * height,  # Alto de la caja.
            linewidth=1,
            edgecolor="r",  # Color de borde rojo.
            facecolor="none",  # Sin relleno.
        )
        # Añadir el parche al eje.
        ax.add_patch(rect)

    plt.show()  # Mostrar la imagen con las cajas.


def get_bboxes(
    loader,
    model,
    iou_threshold,
    threshold,
    pred_format="cells",
    box_format="midpoint",
    device="cuda",
):
    """
    Obtiene las predicciones de cajas delimitadoras y las cajas reales (ground truth)
    de un modelo de detección de objetos a partir de un conjunto de datos cargado en 'loader'.

    Parámetros:
        loader (torch DataLoader): DataLoader que contiene el conjunto de datos.
        model (torch.nn.Module): El modelo de detección de objetos entrenado.
        iou_threshold (float): Umbral para el cálculo de IoU en la supresión no máxima (NMS).
        threshold (float): Umbral para descartar predicciones con baja confianza.
        pred_format (str): Formato de las predicciones ("cells" para YOLO basado en celdas).
        box_format (str): Formato de las cajas ("midpoint" o "corners").
        device (str): Dispositivo donde se ejecutan las operaciones ("cuda" o "cpu").

    Retorna:
        (list, list): Listas de las cajas predichas y las cajas reales.
    """

    # Listas para almacenar las cajas predichas y las reales.
    all_pred_boxes = []
    all_true_boxes = []

    # Coloca el modelo en modo evaluación (eval).
    model.eval()
    train_idx = 0  # Índice para la imagen actual dentro del lote de entrenamiento.

    # Recorre el conjunto de datos cargado en el DataLoader.
    for batch_idx, (x, labels) in enumerate(loader):
        x = x.to(device)  # Mueve las imágenes a la GPU o CPU.
        labels = labels.to(device)  # Mueve las etiquetas a la GPU o CPU.

        # Desactiva el cálculo de gradientes ya que estamos en evaluación.
        with torch.no_grad():
            predictions = model(x)  # Obtiene las predicciones del modelo.

        batch_size = x.shape[0]  # Tamaño del lote.
        true_bboxes = cellboxes_to_boxes(
            labels
        )  # Convierte las cajas reales al formato adecuado.
        bboxes = cellboxes_to_boxes(
            predictions
        )  # Convierte las predicciones al formato adecuado.

        # Para cada imagen en el lote:
        for idx in range(batch_size):
            # Aplica Non-Maximum Suppression (NMS) para filtrar las predicciones superpuestas.
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format=box_format,
            )

            # Almacena las predicciones procesadas en la lista `all_pred_boxes`.
            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            # Almacena las cajas reales (ground truth) en `all_true_boxes`.
            for box in true_bboxes[idx]:
                if (
                    box[1] > threshold
                ):  # Solo considera cajas reales con alta confianza.
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1  # Incrementa el índice de la imagen.

    # Vuelve a colocar el modelo en modo de entrenamiento (train).
    model.train()

    return (
        all_pred_boxes,
        all_true_boxes,
    )  # Retorna las listas de predicciones y cajas reales.


def convert_cellboxes(predictions, S=7):
    """
    Convierte las cajas delimitadoras de las predicciones de YOLO, que están en relación
    con las celdas de la imagen, a coordenadas relativas a la imagen completa.

    Parámetros:
        predictions (torch.Tensor): Predicciones del modelo YOLO en formato de celdas.
        S (int): Tamaño de la cuadrícula en la que se divide la imagen (SxS).

    Retorna:
        torch.Tensor: Cajas delimitadoras convertidas al formato de coordenadas relativas a la imagen.
    """

    predictions = predictions.to("cpu")  # Mueve las predicciones a la CPU.
    batch_size = predictions.shape[0]  # Tamaño del lote.
    predictions = predictions.reshape(
        batch_size, S, S, 30
    )  # Redimensiona las predicciones.

    # Extrae las cajas delimitadoras propuestas por las dos cajas ancla (boxes1 y boxes2).
    bboxes1 = predictions[..., 21:25]
    bboxes2 = predictions[..., 26:30]

    # Concatenar las puntuaciones de las dos cajas ancla para elegir la mejor.
    scores = torch.cat(
        (predictions[..., 20].unsqueeze(0), predictions[..., 25].unsqueeze(0)), dim=0
    )

    # Determina cuál de las dos cajas ancla tiene la mejor puntuación.
    best_box = scores.argmax(0).unsqueeze(-1)

    # Selecciona las mejores cajas delimitadoras entre las dos opciones.
    best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2

    # Calcula los índices de las celdas para ajustar las coordenadas.
    cell_indices = torch.arange(S).repeat(batch_size, S, 1).unsqueeze(-1)

    # Convierte las coordenadas x e y de las celdas al formato de la imagen completa.
    x = 1 / S * (best_boxes[..., :1] + cell_indices)
    y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))

    # Calcula el ancho y alto de las cajas relativo a la imagen.
    w_y = 1 / S * best_boxes[..., 2:4]

    # Concatena las coordenadas convertidas (x, y, ancho, alto).
    converted_bboxes = torch.cat((x, y, w_y), dim=-1)

    # Predice la clase más probable y la mejor confianza.
    predicted_class = predictions[..., :20].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(predictions[..., 20], predictions[..., 25]).unsqueeze(
        -1
    )

    # Concatena la clase predicha, la confianza y las coordenadas de las cajas.
    converted_preds = torch.cat(
        (predicted_class, best_confidence, converted_bboxes), dim=-1
    )

    return converted_preds  # Retorna las predicciones convertidas.


def cellboxes_to_boxes(out, S=7):
    """
    Convierte las predicciones de celdas de YOLO en cajas delimitadoras formateadas
    en coordenadas relativas a la imagen completa.

    Parámetros:
        out (torch.Tensor): Tensor de predicciones de salida del modelo.
        S (int): Tamaño de la cuadrícula (SxS) en la que se divide la imagen.

    Retorna:
        list: Lista de cajas delimitadoras convertidas para todas las imágenes en el lote.
    """

    # Convierte las predicciones en coordenadas de la imagen completa.
    converted_pred = convert_cellboxes(out).reshape(out.shape[0], S * S, -1)

    # Convierte las clases predichas a enteros.
    converted_pred[..., 0] = converted_pred[..., 0].long()

    all_bboxes = []  # Lista para almacenar todas las cajas de todas las imágenes.

    # Para cada imagen en el lote:
    for ex_idx in range(out.shape[0]):
        bboxes = []  # Lista para almacenar las cajas de una sola imagen.
        # Para cada celda en la cuadrícula SxS:
        for bbox_idx in range(S * S):
            # Convierte las coordenadas de cada caja a formato de lista y las almacena.
            bboxes.append([x.item() for x in converted_pred[ex_idx, bbox_idx, :]])

        # Almacena las cajas de la imagen actual en la lista principal.
        all_bboxes.append(bboxes)

    return all_bboxes  # Retorna las cajas de todas las imágenes.


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    """
    Guarda un punto de control (checkpoint) que incluye el estado del modelo y del optimizador.

    Parámetros:
        state (dict): Un diccionario que contiene el estado del modelo y del optimizador.
        filename (str): El nombre del archivo donde se guardará el checkpoint.

    Retorna:
        None
    """

    print("=> Saving checkpoint")
    torch.save(
        state, filename
    )  # Guarda el estado del modelo y optimizador en un archivo.


def load_checkpoint(checkpoint, model, optimizer):
    """
    Carga un punto de control (checkpoint) previamente guardado que incluye el estado
    del modelo y del optimizador.

    Parámetros:
        checkpoint (dict): El punto de control guardado que contiene el estado del modelo y del optimizador.
        model (torch.nn.Module): El modelo de red neuronal al que se cargará el estado.
        optimizer (torch.optim.Optimizer): El optimizador al que se cargará el estado.

    Retorna:
        None
    """

    print("=> Loading checkpoint")
    # Carga el estado del modelo y el optimizador desde el punto de control.
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
