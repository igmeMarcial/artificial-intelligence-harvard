""" Se inicializan las configuraciones y los hiperparámetros.
Se preparan los datos, se definen las transformaciones y se configuran los cargadores de datos.
El modelo se entrena por varias épocas utilizando la función de pérdida YOLO y el optimizador Adam.
Se calculan métricas como la mAP para evaluar el rendimiento del modelo."""

import torch  # Biblioteca principal de PyTorch
import torchvision.transforms as transforms  # Transformaciones de imágenes de Torchvision
import torch.optim as optim  # Optimizadores en PyTorch
import torchvision.transforms.functional as FT  # Funciones de transformaciones de Torchvision
from tqdm import tqdm  # Muestra barras de progreso para bucles
from torch.utils.data import DataLoader  # Herramienta para cargar datos en lotes
from model import Yolov1  # Importa la clase del modelo YOLOv1
from dataset import VOCDataset  # Dataset para el conjunto de datos Pascal VOC
from utils import (  # Funciones utilitarias para ayudar en la detección de objetos y entrenamiento
    non_max_suppression,  # Supresión no máxima para eliminar detecciones redundantes
    mean_average_precision,  # Calcula la media de la precisión promedio (mAP)
    intersection_over_union,  # Calcula la Intersección sobre la Unión (IoU)
    cellboxes_to_boxes,  # Convierte cajas en formato de celdas a cajas de detección
    get_bboxes,  # Obtiene las cajas delimitadoras predichas y reales
    plot_image,  # Dibuja las imágenes con las cajas delimitadoras
    save_checkpoint,  # Guarda el estado actual del modelo en un checkpoint
    load_checkpoint,  # Carga un checkpoint guardado previamente
)
from loss import YoloLoss  # Función de pérdida específica para YOLO

# Fija una semilla para asegurar reproducibilidad
seed = 123
torch.manual_seed(seed)

# Hiperparámetros y configuraciones adicionales
LEARNING_RATE = 2e-5  # Tasa de aprendizaje para el optimizador
DEVICE = (
    "cuda" if torch.cuda.is_available() else "cpu"
)  # Utiliza GPU si está disponible, si no CPU
BATCH_SIZE = 16  # Tamaño de lote (16 ejemplos por lote)
WEIGHT_DECAY = 0  # Regularización L2 (aquí es 0, no se usa)
EPOCHS = 1000  # Número de épocas de entrenamiento
NUM_WORKERS = 2  # Número de hilos para cargar los datos en paralelo
PIN_MEMORY = True  # Mejora la transferencia de datos a la GPU
LOAD_MODEL = False  # Indica si se cargará un modelo previamente guardado
LOAD_MODEL_FILE = (
    "overfit.pth.tar"  # Archivo de modelo guardado a cargar (si LOAD_MODEL es True)
)
IMG_DIR = "data/images"  # Directorio de imágenes
LABEL_DIR = "data/labels"  # Directorio de etiquetas


# Clase Compose para aplicar múltiples transformaciones a las imágenes y sus cajas delimitadoras
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms  # Guarda una lista de transformaciones

    def __call__(self, img, bboxes):
        # Aplica cada transformación en la imagen, pero deja las cajas (bboxes) sin cambios
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes  # Retorna la imagen transformada y las cajas sin cambios


# Definición de las transformaciones a aplicar
transform = Compose(
    [
        transforms.Resize((448, 448)),  # Redimensiona las imágenes a 448x448 píxeles
        transforms.ToTensor(),
    ]
)  # Convierte las imágenes en tensores (para PyTorch)


# Función para entrenar el modelo en un lote de datos
def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(
        train_loader, leave=True
    )  # Barra de progreso sobre los datos del loader
    mean_loss = []  # Lista para almacenar la pérdida media por lote

    # Itera sobre el DataLoader
    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(
            DEVICE
        )  # Mueve los datos (imágenes y etiquetas) a la GPU/CPU
        out = model(
            x
        )  # Hace un forward pass a través del modelo con las imágenes de entrada
        loss = loss_fn(
            out, y
        )  # Calcula la pérdida entre las predicciones y las etiquetas
        mean_loss.append(loss.item())  # Almacena el valor de la pérdida en la lista

        optimizer.zero_grad()  # Limpia los gradientes acumulados en el optimizador
        loss.backward()  # Calcula los gradientes de la pérdida mediante backpropagation
        optimizer.step()  # Actualiza los pesos del modelo con los gradientes calculados

        # Actualiza la barra de progreso con la pérdida actual
        loop.set_postfix(loss=loss.item())

    # Imprime la pérdida promedio del lote al final de la época
    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")


# Función principal donde ocurre el entrenamiento
def main():
    # Inicializa el modelo YOLOv1 con los parámetros (tamaño de la celda, número de cajas, clases)
    model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    # Configura el optimizador Adam para ajustar los parámetros del modelo
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    # Define la función de pérdida (YoloLoss)
    loss_fn = YoloLoss()

    # Carga el modelo desde un checkpoint si LOAD_MODEL es True
    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

    # Crea el conjunto de datos de entrenamiento a partir de un archivo CSV
    train_dataset = VOCDataset(
        "data/100examples.csv",  # Ruta del archivo de entrenamiento
        transform=transform,  # Aplicar transformaciones
        img_dir=IMG_DIR,  # Directorio de las imágenes
        label_dir=LABEL_DIR,  # Directorio de las etiquetas
    )

    # Crea el conjunto de datos de prueba
    test_dataset = VOCDataset(
        "data/test.csv",  # Ruta del archivo de prueba
        transform=transform,  # Aplicar transformaciones
        img_dir=IMG_DIR,  # Directorio de las imágenes
        label_dir=LABEL_DIR,  # Directorio de las etiquetas
    )

    # Cargador de datos para el entrenamiento
    train_loader = DataLoader(
        dataset=train_dataset,  # El dataset de entrenamiento
        batch_size=BATCH_SIZE,  # El tamaño de lote
        num_workers=NUM_WORKERS,  # Número de hilos para cargar los datos
        pin_memory=PIN_MEMORY,  # Para acelerar la transferencia de datos a GPU
        shuffle=True,  # Barajar los datos en cada época
        drop_last=True,  # Descarta el último lote si tiene menos de BATCH_SIZE
    )

    # Cargador de datos para la prueba
    test_loader = DataLoader(
        dataset=test_dataset,  # El dataset de prueba
        batch_size=BATCH_SIZE,  # El tamaño de lote
        num_workers=NUM_WORKERS,  # Número de hilos para cargar los datos
        pin_memory=PIN_MEMORY,  # Para acelerar la transferencia de datos a GPU
        shuffle=True,  # Barajar los datos en cada época
        drop_last=True,  # Descarta el último lote si tiene menos de BATCH_SIZE
    )

    # Bucle principal de entrenamiento durante las épocas definidas
    for epoch in range(EPOCHS):
        # Obtiene las cajas predichas y las cajas reales (objetivo) usando IoU
        pred_boxes, target_boxes = get_bboxes(
            train_loader, model, iou_threshold=0.5, threshold=0.4
        )

        # Calcula la precisión promedio (mAP) de las predicciones
        mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
        )
        print(f"Train mAP: {mean_avg_prec}")  # Imprime la mAP del entrenamiento

        # Llama a la función de entrenamiento para cada época
        train_fn(train_loader, model, optimizer, loss_fn)


# Ejecuta la función main si el archivo es ejecutado directamente
if __name__ == "__main__":
    main()
