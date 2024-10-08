"""
Implementation of YOLOv3 architecture
"""

# Importamos las herramientas de PyTorch para construir redes neuronales
import torch
import torch.nn as nn  # nn para crear redes neuronales como convulucionales ,d ensas , de pooling etc

""" 
Information about architecture config:
Tuple is structured by (filters, kernel_size, stride) 
Every conv is a same convolution. 
List is structured by "B" indicating a residual block followed by the number of repeats
"S" is for scale prediction block and computing the yolo loss
"U" is for upsampling the feature map and concatenating with a previous layer
"""
config = [
    (32, 3, 1),  # Convolución con 32 filtros, tamaño de kernel 3x3, stride 1
    (
        64,
        3,
        2,
    ),  # Convolución con 64 filtros, tamaño de kernel 3x3, stride 2 (reduce la resolución a la mitad)
    ["B", 1],  # Bloque residual (B) que se repite 1 vez
    (128, 3, 2),  # Convolución con 128 filtros, tamaño de kernel 3x3, stride 2
    ["B", 2],  # Bloque residual (B) que se repite 2 veces
    (256, 3, 2),  # Convolución con 256 filtros, tamaño de kernel 3x3, stride 2
    ["B", 8],  # Bloque residual (B) que se repite 8 veces
    (512, 3, 2),  # Convolución con 512 filtros, tamaño de kernel 3x3, stride 2
    ["B", 8],  # Bloque residual (B) que se repite 8 veces
    (1024, 3, 2),  # Convolución con 1024 filtros, tamaño de kernel 3x3, stride 2
    ["B", 4],  # Bloque residual que se repite 4 veces
    (512, 1, 1),  # Convolución con kernel de 1x1 para reducir la dimensionalidad
    (1024, 3, 1),  # Convolución con kernel de 3x3
    "S",  # Bloque de predicción de escala
    (256, 1, 1),  # Convolución con kernel de 1x1 (reduce el tamaño de la salida)
    "U",  # Upsampling (duplicar la resolución)
    (256, 1, 1),  # Convolución con kernel de 1x1 después del 'upsampling'
    (512, 3, 1),  # Convolución con kernel de 3x3
    "S",  # Otro bloque de predicción de escala
    (128, 1, 1),  # Convolución con kernel de 1x1
    "U",  # Otro upsampling
    (128, 1, 1),  # Convolución con kernel de 1x1
    (256, 3, 1),  # Convolución con kernel de 3x3
    "S",  # Bloque final de predicción de escala
]


# CNNBlock es un bloque básico de convolución que incluye convolución, batch normalization y activación LeakyReLU.


class CNNBlock(nn.Module):  # Clase que define un bloque de convolución
    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
        # Capa de convolución 2D (con o sin batch normalization y activación)
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs)
        self.bn = nn.BatchNorm2d(
            out_channels
        )  # Batch normalization para estabilizar el entrenamiento
        self.leaky = nn.LeakyReLU(
            0.1
        )  # Activación Leaky ReLU para introducir no linealidades
        self.use_bn_act = bn_act  # Controla si usamos batch normalization y activación

    def forward(self, x):
        if self.use_bn_act:
            return self.leaky(
                self.bn(self.conv(x))
            )  # Si bn_act es True, aplicamos convolución, batch norm y activación
        else:
            return self.conv(x)  # Si bn_act es False, solo aplicamos convolución


""" 
ResidualBlock define un bloque residual que puede repetirse varias veces.
Usa "conexiones residuales", donde la salida de una capa se suma a la entrada original 
(esto ayuda a que las redes profundas aprendan mejor).
"""


class ResidualBlock(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super().__init__()
        self.layers = nn.ModuleList()  # Crea una lista de módulos de capas
        for repeat in range(num_repeats):  # Por cada repetición
            # Añade dos bloques CNN: uno con kernel de 1x1 y otro con kernel de 3x3
            self.layers += [
                nn.Sequential(
                    CNNBlock(
                        channels, channels // 2, kernel_size=1
                    ),  # Convolución para reducir los canales
                    CNNBlock(
                        channels // 2, channels, kernel_size=3, padding=1
                    ),  # Convolución para devolver el tamaño original
                )
            ]

        self.use_residual = (
            use_residual  # Si True, aplicamos conexión residual (skip connection)
        )
        self.num_repeats = num_repeats  # Cuántas veces se repite el bloque residual

    def forward(self, x):
        for layer in self.layers:  # Para cada capa en el bloque residual
            if self.use_residual:
                x = x + layer(
                    x
                )  # Si usamos residual, sumamos la salida con la entrada original (skip connection)
            else:
                x = layer(
                    x
                )  # Si no usamos residual, solo pasamos la entrada por la capa

        return x  # Devuelve la salida final


""" ScalePrediction se usa para generar las predicciones de las cajas delimitadoras (bounding boxes).
La salida incluye información sobre las coordenadas de las cajas, la confianza y las clases detectadas."""


class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pred = nn.Sequential(
            CNNBlock(
                in_channels, 2 * in_channels, kernel_size=3, padding=1
            ),  # Aumenta el número de canales
            CNNBlock(
                2 * in_channels, (num_classes + 5) * 3, bn_act=False, kernel_size=1
            ),  # Conv final para predicciones
        )
        self.num_classes = num_classes  # Número de clases a predecir

    def forward(self, x):
        return (
            self.pred(x)  # Pasa la entrada por las capas CNN
            .reshape(
                x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3]
            )  # Ajusta el tensor de salida
            .permute(
                0, 1, 3, 4, 2
            )  # Cambia el orden de las dimensiones para que sea compatible con YOLO
        )


class YOLOv3(nn.Module):
    def __init__(self, in_channels=3, num_classes=80):
        super().__init__()
        self.num_classes = num_classes  # El número de clases que queremos predecir (por ejemplo, 80 clases en COCO)
        self.in_channels = in_channels  # Los canales de entrada de la imagen (por defecto, 3 para imágenes RGB
        self.layers = (
            self._create_conv_layers()
        )  # Crea las capas convolucionales a partir del config que definimos antes

    def forward(self, x):
        outputs = []  # Para guardar las predicciones de las 3 escalas
        route_connections = []  # Para las conexiones entre capas (skip connections)
        for layer in self.layers:  # Recorremos cada capa en las capas del modelo
            if isinstance(
                layer, ScalePrediction
            ):  # Si la capa es una predicción de escala
                outputs.append(
                    layer(x)
                )  # Añade la salida a 'outputs' (las predicciones de la capa)
                continue  # Continua con la siguiente capa

            x = layer(x)  # Si no es predicción, pasa la entrada (x) a través de la capa

            if (
                isinstance(layer, ResidualBlock) and layer.num_repeats == 8
            ):  # Si es un bloque residual con 8 repeticiones
                route_connections.append(
                    x
                )  # Guarda la salida para hacer una conexión más adelante

            elif isinstance(
                layer, nn.Upsample
            ):  # Si es una capa de 'upsampling' (aumentar el tamaño de la imagen)
                x = torch.cat(
                    [x, route_connections[-1]], dim=1
                )  # Concatenamos la salida con una anterior (skip connection)
                route_connections.pop()  # Eliminamos esa conexión de la lista
        return outputs  # Devuelve las predicciones de las tres escalas (outputs)

    def _create_conv_layers(self):
        layers = nn.ModuleList()  # Lista de capas del modelo
        in_channels = self.in_channels  # Canales de entrada, inicialmente 3 (RGB)

        for (
            module
        ) in config:  # Recorremos cada módulo en la configuración (la lista config)
            if isinstance(
                module, tuple
            ):  # Si es una tupla (definición de una capa convolucional)
                out_channels, kernel_size, stride = (
                    module  # Desempaquetamos los valores
                )
                layers.append(
                    CNNBlock(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=(
                            1 if kernel_size == 3 else 0
                        ),  # Padding solo si el kernel es 3x3
                    )
                )
                in_channels = out_channels  # Actualizamos los canales de entrada para la siguiente capa

            elif isinstance(module, list):  # Si es una lista (bloque residual)
                num_repeats = module[1]
                layers.append(
                    ResidualBlock(
                        in_channels,
                        num_repeats=num_repeats,  # Añadimos el bloque residual al modelo
                    )
                )

            elif isinstance(module, str):  # Si es un string ("S" o "U")
                if module == "S":  # Si es "S" (bloque de predicción de escala)
                    layers += [
                        ResidualBlock(
                            in_channels, use_residual=False, num_repeats=1
                        ),  # Bloque residual
                        CNNBlock(
                            in_channels, in_channels // 2, kernel_size=1
                        ),  # Capa convolucional para reducir los canales
                        ScalePrediction(
                            in_channels // 2, num_classes=self.num_classes
                        ),  # Capa de predicción de escala
                    ]
                    in_channels = (
                        in_channels // 2
                    )  # Reducimos los canales de salida para las siguientes capas

                elif module == "U":  # Si es "U" (upsampling)
                    layers.append(
                        nn.Upsample(
                            scale_factor=2
                        ),  # Añadimos una capa de upsampling (duplica la resolución)
                    )
                    in_channels = (
                        in_channels * 3
                    )  # Aumentamos el número de canales después de concatenar

        return layers


if __name__ == "__main__":
    num_classes = (
        20  # Número de clases para el conjunto de datos (20 para este ejemplo)
    )
    IMAGE_SIZE = 416  # Tamaño de la imagen de entrada (416x416 píxeles)
    model = YOLOv3(num_classes=num_classes)  # Instanciamos el modelo YOLOv3
    x = torch.randn(
        (2, 3, IMAGE_SIZE, IMAGE_SIZE)
    )  # Creamos una entrada simulada (batch de 2 imágenes)
    out = model(x)  # Pasamos la entrada por el modelo
    assert model(x)[
        0
    ].shape == (  # Comprobamos que la salida para la primera escala sea correcta
        2,
        3,
        IMAGE_SIZE // 32,
        IMAGE_SIZE // 32,
        num_classes + 5,
    )
    assert model(x)[1].shape == (  # Comprobamos la segunda escala
        2,
        3,
        IMAGE_SIZE // 16,
        IMAGE_SIZE // 16,
        num_classes + 5,
    )
    assert model(x)[2].shape == (  # Comprobamos la tercera escala
        2,
        3,
        IMAGE_SIZE // 8,
        IMAGE_SIZE // 8,
        num_classes + 5,
    )
    print("Success!")  # Si todo va bien, imprime "Success!"

    """ YOLOv3 es un modelo que toma imágenes de entrada y predice objetos a diferentes escalas.
Usa bloques residuales, convoluciones, upsampling y predicciones de escala.
El método de prueba al final del código asegura que el modelo funcione correctamente con entradas simuladas."""
