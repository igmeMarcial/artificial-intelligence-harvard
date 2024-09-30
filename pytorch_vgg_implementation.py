# Este código define y construye una red neuronal convolucional basada en la arquitectura VGG16
import torch  # entrenar redes neuronales.
import torch.nn as nn  # Módulo para crear capas de redes neuronales
import torch.optim as optim  # Módulo para optimizadores (entrenamiento)
import torch.nn.functional as F  # Funciones como activaciones y operaciones de red
import torch.utils.data as DataLoader  # Para manejar datos y minibatches
import torchvision.datasets as datasets  # Conjuntos de datos comunes como CIFAR, MNIST
import torchvision.transforms as transforms  # Transformaciones de imágenes como recorte, normalización


VGG16 = [
    64,  # 2 capas convolucionales con 64 filtros, seguidas de max pooling.
    64,
    "M",
    128,  # 2 capas convolucionales con 128 filtros, seguidas de max pooling.
    128,
    "M",
    256,  # 3 capas convolucionales con 256 filtros, seguidas de max pooling.
    256,
    256,
    "M",
    512,  # 3 capas convolucionales con 512 filtros, seguidas de max pooling.
    512,
    512,
    "M",
    512,  # 3 capas convolucionales con 512 filtros, seguidas de max pooling.
    512,
    512,
    "M",
]

# VGG16 = [...]: Esta lista define la arquitectura de la red VGG16, donde los números representan la cantidad de filtros en las capas convolucionales y 'M' representa una operación de max pooling. La arquitectura es la siguiente:
# 64, 64: Dos capas convolucionales con 64 filtros.
# 'M': Max pooling, que reduce las dimensiones espaciales de las características.
# 128, 128: Dos capas convolucionales con 128 filtros.
# 'M': Max pooling.
# 256, 256, 256: Tres capas convolucionales con 256 filtros.
# 'M': Max pooling.
# 512, 512, 512: Tres capas convolucionales con 512 filtros.
# 'M': Max pooling.
# 512, 512, 512: Tres capas convolucionales con 512 filtros.
# 'M': Max pooling.


class VGG_net(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(VGG_net, self).__init__()
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layers(
            VGG16
        )  # Crear las capas convolucionales basadas en la arquitectura VGG16.
        self.fcs = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),  # Capa lineal con 4096 neuronas de salida.
            nn.ReLU(),  # Función de activación ReLU.
            nn.Dropout(
                p=0.5
            ),  # Dropout para prevenir el sobreajuste (50% de desconexión aleatoria de neuronas).
            nn.Linear(
                4096, num_classes
            ),  # Última capa lineal para clasificar en num_classes clases.
        )

    def forward(self, x):
        x = self.conv_layers(
            x
        )  # Pasar la entrada a través de las capas convolucionales.
        x = x.reshape(x.shape[0], -1)  # Aplanar el tensor a un vector.
        x = self.fcs(x)  # Pasar el vector a través de las capas lineales.
        return x

    def create_conv_layers(self, architecture):
        layers = []  # Lista para almacenar las capas.
        in_channels = self.in_channels  # Canales de entrada.

        for x in architecture:
            if type(x) == int:
                out_channels = x
                layers += [
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=(1, 1),
                    ),  # Capa convolucional con filtros de 3x3
                    nn.BatchNorm2d(
                        x
                    ),  # Normalización de Batch para mejorar la estabilidad.
                    nn.ReLU(),  # Función de activación ReLU.
                ]
                in_channels = (
                    x  # Actualiza los canales de entrada para la siguiente capa.
                )
            elif x == "M":
                layers += [
                    nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
                ]  # Operación de max pooling para reducir la dimensión espacial.
        return nn.Sequential(
            *layers
        )  # Combina todas las capas en un bloque secuencial.


device = "cuda" if torch.cuda.is_available() else "cpu"
# Usa GPU si está disponible, de lo contrario, usa CPU.
model = VGG_net(in_channels=3, num_classes=1000).to(
    device
)  # Crea una instancia del modelo VGG y lo envía al dispositivo.
x = torch.randn(1, 3, 224, 224).to(
    device
)  # Crea un tensor aleatorio que simula una imagen de entrada de tamaño 224x224 (1 imagen, 3 canales).
print(model(x).shape)  # Pasa la imagen por el modelo y muestra la forma de la salida.
