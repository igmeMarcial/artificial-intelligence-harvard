# Este código prepara el entorno para entrenar un modelo de detección de objetos con YOLO.
import albumentations as A  # Importa Albumentations para las transformaciones de datos de imagen.
import cv2  # OpenCV para manipular imágenes.
import torch  # PyTorch para definir y entrenar el modelo.
from albumentations.pytorch import (
    ToTensorV2,
)  # Transforma imágenes a tensores para PyTorch.
from utils import seed_everything  # Función opcional para asegurar reproducibilidad.

# Configuración general
DATASET = "PASCAL_VOC"  # Nombre del dataset que se utilizará.
DEVICE = (
    "cuda" if torch.cuda.is_available() else "cpu"
)  # Usa GPU si está disponible, si no, usa CPU.
# seed_everything()  # (Opcional) Inicializa la semilla para tener resultados deterministas.
NUM_WORKERS = 4  # Número de trabajadores (threads) para cargar los datos.
BATCH_SIZE = 32  # Tamaño de lote durante el entrenamiento.
IMAGE_SIZE = 416  # Tamaño de las imágenes que se usarán.
NUM_CLASSES = 20  # Número de clases en el dataset Pascal VOC (20 en total).
LEARNING_RATE = 1e-5  # Tasa de aprendizaje para optimizar el modelo.
WEIGHT_DECAY = 1e-4  # Decaimiento del peso para regularización L2.
NUM_EPOCHS = 100  # Número de épocas de entrenamiento.
CONF_THRESHOLD = 0.05  # Umbral de confianza para las predicciones de objeto.
MAP_IOU_THRESH = 0.5  # Umbral de IoU para medir mAP (Mean Average Precision).
NMS_IOU_THRESH = 0.45  # Umbral de IoU para la Supresión de No Máximos (NMS).
S = [
    IMAGE_SIZE // 32,
    IMAGE_SIZE // 16,
    IMAGE_SIZE // 8,
]  # Tamaños de grilla para diferentes escalas.
PIN_MEMORY = True  # Activa la optimización de memoria si está disponible.
LOAD_MODEL = True  # Indica si se cargará un modelo preentrenado.
SAVE_MODEL = True  # Indica si se guardará el modelo después de entrenar.
CHECKPOINT_FILE = (
    "checkpoint.pth.tar"  # Nombre del archivo para guardar el checkpoint del modelo.
)
IMG_DIR = DATASET + "/images/"  # Directorio donde se encuentran las imágenes.
LABEL_DIR = DATASET + "/labels/"  # Directorio donde se encuentran las etiquetas.

# Anclas predefinidas (anchors) para diferentes escalas.
ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],  # Anclas para la escala más grande.
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],  # Anclas para la escala media.
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],  # Anclas para la escala más pequeña.
]  # Todos los valores están entre [0, 1] ya que se normalizan respecto al tamaño de la imagen.

# Escala para las transformaciones de imagen.
scale = 1.1

# Transformaciones para el conjunto de entrenamiento (train_transforms).
train_transforms = A.Compose(
    [
        A.LongestMaxSize(
            max_size=int(IMAGE_SIZE * scale)
        ),  # Ajusta la imagen para que su lado más largo tenga el tamaño indicado.
        A.PadIfNeeded(
            min_height=int(IMAGE_SIZE * scale),
            min_width=int(IMAGE_SIZE * scale),
            border_mode=cv2.BORDER_CONSTANT,  # Añade relleno si es necesario.
        ),
        A.RandomCrop(
            width=IMAGE_SIZE, height=IMAGE_SIZE
        ),  # Realiza un recorte aleatorio de la imagen.
        A.ColorJitter(
            brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=0.4
        ),  # Ajusta el color, brillo, contraste, etc., de manera aleatoria.
        A.OneOf(
            [
                A.ShiftScaleRotate(
                    rotate_limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT
                ),  # Aplica un desplazamiento, escalado y rotación aleatorios.
                A.IAAAffine(
                    shear=15, p=0.5, mode="constant"
                ),  # Realiza una transformación affine con cizallamiento.
            ],
            p=1.0,
        ),
        A.HorizontalFlip(
            p=0.5
        ),  # Voltea la imagen horizontalmente de manera aleatoria.
        A.Blur(p=0.1),  # Aplica desenfoque con una probabilidad del 10%.
        A.CLAHE(p=0.1),  # Aplicación de histogram equalization (realza el contraste).
        A.Posterize(p=0.1),  # Reduce la cantidad de colores de la imagen.
        A.ToGray(p=0.1),  # Convierte la imagen a escala de grises.
        A.ChannelShuffle(p=0.05),  # Reorganiza los canales de color de la imagen.
        A.Normalize(
            mean=[0, 0, 0],
            std=[1, 1, 1],
            max_pixel_value=255,
        ),  # Normaliza la imagen.
        ToTensorV2(),  # Convierte la imagen y las cajas a tensores para PyTorch.
    ],
    bbox_params=A.BboxParams(
        format="yolo", min_visibility=0.4, label_fields=[]
    ),  # Parámetros para las cajas delimitadoras.
)

# Transformaciones para el conjunto de prueba (test_transforms).
test_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=IMAGE_SIZE),  # Ajusta el tamaño de la imagen.
        A.PadIfNeeded(
            min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT
        ),  # Añade relleno si es necesario.
        A.Normalize(
            mean=[0, 0, 0],
            std=[1, 1, 1],
            max_pixel_value=255,
        ),  # Normaliza la imagen.
        ToTensorV2(),  # Convierte la imagen a tensor.
    ],
    bbox_params=A.BboxParams(
        format="yolo", min_visibility=0.4, label_fields=[]
    ),  # Parámetros para las cajas delimitadoras.
)

# Lista de clases del dataset Pascal VOC.
PASCAL_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

# Lista de etiquetas de clases para el dataset COCO.
COCO_LABELS = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]
