"""
Implementación de la función de pérdida de Yolo similar a la del artículo de Yolov3.
La diferencia principal es que se usa CrossEntropy para las clases en lugar de BinaryCrossEntropy.
"""

import random
import torch
import torch.nn as nn

from utils import (
    intersection_over_union,
)  # Importa la función para calcular la Intersección sobre Unión (IoU), importante para medir qué tan cerca están las predicciones de los bounding boxes reales.


class YoloLoss(nn.Module):

    def __init__(self):
        super().__init__()  # Inicializa la clase padre (nn.Module).
        self.mse = (
            nn.MSELoss()
        )  # Mean Squared Error (MSE) para calcular la pérdida de las coordenadas de la caja.
        self.bce = (
            nn.BCEWithLogitsLoss()
        )  # Binary Cross Entropy con Logits para la predicción de la presencia de objetos.
        self.entropy = (
            nn.CrossEntropyLoss()
        )  # CrossEntropy para la clasificación de clases.
        self.sigmoid = (
            nn.Sigmoid()
        )  # Función sigmoide para transformar las predicciones en probabilidades entre 0 y 1.

        # Constantes que ponderan las diferentes partes de la pérdida.
        self.lambda_class = 1  # Pondera la pérdida de clasificación (class loss).
        self.lambda_noobj = (
            10  # Pondera la pérdida cuando no hay objeto (no object loss).
        )
        self.lambda_obj = 1  # Pondera la pérdida cuando hay un objeto (object loss).
        self.lambda_box = 10  # Pondera la pérdida de las coordenadas de la caja delimitadora (bounding box loss).

    def forward(
        self, predictions, target, anchors
    ):  # Método que realiza el cálculo de la pérdida.
        # Check where obj and noobj (we ignore if target == -1)
        obj = target[..., 0] == 1  # Marca las posiciones donde hay un objeto (Iobj_i).
        noobj = (
            target[..., 0] == 0
        )  # Marca las posiciones donde no hay objeto (Inoobj_i).

        # ======================= #
        # PÉRDIDA PARA NO OBJETOS #
        # ======================= #

        no_object_loss = self.bce(
            (predictions[..., 0:1][noobj]),
            (target[..., 0:1][noobj]),
        )

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #

        anchors = anchors.reshape(
            1, 3, 1, 1, 2
        )  # Da forma a los anclajes para que coincidan con las predicciones.
        # Predecir coordenadas x, y (con sigmoide) y ajustar ancho y alto (log transformado) usando los anclajes.
        box_preds = torch.cat(
            [
                self.sigmoid(
                    predictions[..., 1:3]
                ),  # Aplica la sigmoide a las coordenadas x, y.
                torch.exp(predictions[..., 3:5])
                * anchors,  # Usa exponencial para predecir el ancho y alto de la caja.
            ],
            dim=-1,
        )
        # Calcula la IoU (Intersección sobre Unión) entre las predicciones y las cajas reales.
        ious = intersection_over_union(box_preds[obj], target[..., 1:5][obj]).detach()
        # Calcula la pérdida para las celdas con objeto, ponderada por el IoU.
        object_loss = self.mse(
            self.sigmoid(predictions[..., 0:1][obj]), ious * target[..., 0:1][obj]
        )

        # ======================== #
        #   PÉRDIDA PARA COORDENADAS DE CAJAS   #
        # ======================== #

        predictions[..., 1:3] = self.sigmoid(
            predictions[..., 1:3]
        )  # Aplica la sigmoide a las coordenadas x, y.
        # Ajusta el ancho y alto para que sean relativos a los anclajes, usando logaritmo.
        target[..., 3:5] = torch.log(
            (1e-16 + target[..., 3:5] / anchors)  # Evita divisiones por cero.
        )  # Ancho y alto ajustados.
        # Calcula la pérdida MSE para las celdas con objeto, comparando predicciones y cajas reales.
        box_loss = self.mse(predictions[..., 1:5][obj], target[..., 1:5][obj])

        # ================== #
        #   PERDIDA DE CLASE   #
        # ================== #

        class_loss = self.entropy(
            (
                predictions[..., 5:][obj]
            ),  # Predicciones de clases para las celdas con objeto.
            (target[..., 5][obj].long()),  # Clases reales para las celdas con objeto.
        )

        # print("__________________________________")
        # print(self.lambda_box * box_loss)  # Muestra la pérdida de la caja.
        # print(self.lambda_obj * object_loss)  # Muestra la pérdida del objeto.
        # print(self.lambda_noobj * no_object_loss)  # Muestra la pérdida cuando no hay objeto.
        # print(self.lambda_class * class_loss)  # Muestra la pérdida de la clasificación.
        # print("\n")

        # Retorna la pérdida total ponderada según los factores de lambda.
        return (
            self.lambda_box * box_loss  # Pérdida de las coordenadas de la caja.
            + self.lambda_obj * object_loss  # Pérdida de objeto.
            + self.lambda_noobj * no_object_loss  # Pérdida cuando no hay objeto.
            + self.lambda_class * class_loss  # Pérdida de clasificación.
        )


""" Este código optimiza el entrenamiento de un modelo de detección de objetos YOLO al hacer que
el modelo sea más preciso tanto en la detección de
objetos como en la predicción de sus coordenadas."""
