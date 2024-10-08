"""
Creates a Pytorch dataset to load the Pascal VOC & MS COCO datasets
"""

import config
import numpy as np
import os
import pandas as pd
import torch

from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from utils import (
    cells_to_bboxes,
    iou_width_height as iou,
    non_max_suppression as nms,
    plot_image,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True


""" YOLODataset es una clase que carga imágenes y sus etiquetas (bounding boxes) desde Pascal VOC o MS COCO.
Usa escalas múltiples (13x13, 26x26, 52x52) para predecir objetos de diferentes tamaños.
El método __getitem__ carga cada imagen, aplica transformaciones (si están definidas) y prepara las cajas para diferentes escalas y anclajes.
La función test() evalúa si el dataset y la configuración están correctos, visualizando las predicciones en las imágenes."""


class YOLODataset(Dataset):

    def __init__(
        self,
        csv_file,  # Archivo CSV con las anotaciones (ubicación de imágenes y etiquetas)
        img_dir,  # Directorio donde están las imágenes
        label_dir,  # Directorio donde están los labels (anotaciones de las cajas)
        anchors,  # Los anclajes usados para predecir bounding boxes
        image_size=416,  # Tamaño de la imagen (416x416 para YOLO)
        S=[13, 26, 52],  # Tamaños de las celdas en 3 escalas (diferentes resoluciones)
        C=20,  # Número de clases (20 para Pascal VOC, 80 para COCO)
        transform=None,  # Transformaciones (aumentaciones) aplicadas a las imágenes
    ):
        self.annotations = pd.read_csv(
            csv_file
        )  # Cargamos las anotaciones desde el CSV
        self.img_dir = img_dir  # Directorio de imágenes
        self.label_dir = label_dir  # Directorio de labels
        self.image_size = image_size  # Tamaño de la imagen
        self.transform = transform  # Transformaciones (como aumentar la imagen)
        self.S = S  # Diferentes tamaños de las escalas (S = 13, 26, 52)
        self.anchors = torch.tensor(
            anchors[0] + anchors[1] + anchors[2]
        )  # Convertimos los anclajes en tensores, combinando todas las escalas
        self.num_anchors = self.anchors.shape[0]  # Total de anclajes
        self.num_anchors_per_scale = (
            self.num_anchors // 3
        )  # Número de anclajes por escala
        self.C = C  # Número de clases
        self.ignore_iou_thresh = 0.5  # Umbral de IoU para ignorar predicciones

    def __len__(self):
        return len(self.annotations)  # Retorna el número de imágenes en el dataset

    def __getitem__(self, index):
        """__getitem__ carga una imagen y sus etiquetas (bounding boxes) desde los archivos.
        Carga el label_path de la caja de la imagen y ajusta el formato de los bounding boxes.
        Abre la imagen desde img_path y la convierte en un array de Numpy."""
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        bboxes = np.roll(
            np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1
        ).tolist()
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=image, bboxes=bboxes)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]

        # Below assumes 3 scale predictions (as paper) and same num of anchors per scale
        targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S]
        for box in bboxes:
            iou_anchors = iou(torch.tensor(box[2:4]), self.anchors)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height, class_label = box
            has_anchor = [False] * 3  # each scale should have one anchor
            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                S = self.S[scale_idx]
                i, j = int(S * y), int(S * x)  # which cell
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    x_cell, y_cell = S * x - j, S * y - i  # both between [0,1]
                    width_cell, height_cell = (
                        width * S,
                        height * S,
                    )  # can be greater than 1 since it's relative to cell
                    box_coordinates = torch.tensor(
                        [x_cell, y_cell, width_cell, height_cell]
                    )
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    has_anchor[scale_idx] = True

                elif (
                    not anchor_taken
                    and iou_anchors[anchor_idx] > self.ignore_iou_thresh
                ):
                    targets[scale_idx][
                        anchor_on_scale, i, j, 0
                    ] = -1  # ignore prediction

        return image, tuple(targets)


def test():
    anchors = config.ANCHORS

    transform = config.test_transforms

    dataset = YOLODataset(
        "COCO/train.csv",
        "COCO/images/images/",
        "COCO/labels/labels_new/",
        S=[13, 26, 52],
        anchors=anchors,
        transform=transform,
    )
    S = [13, 26, 52]
    scaled_anchors = torch.tensor(anchors) / (
        1 / torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    )
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    for x, y in loader:
        boxes = []

        for i in range(y[0].shape[1]):
            anchor = scaled_anchors[i]
            print(anchor.shape)
            print(y[i].shape)
            boxes += cells_to_bboxes(
                y[i], is_preds=False, S=y[i].shape[2], anchors=anchor
            )[0]
        boxes = nms(boxes, iou_threshold=1, threshold=0.7, box_format="midpoint")
        print(boxes)
        plot_image(x[0].permute(1, 2, 0).to("cpu"), boxes)


if __name__ == "__main__":
    test()


""" Esta lógica facilita la preparación del dataset, organiza las cajas de anclaje y 
predicciones a diferentes escalas, y maneja el proceso de entrenamiento de manera eficiente,
optimizando el rendimiento del modelo en la tarea de detección de objetos."""
