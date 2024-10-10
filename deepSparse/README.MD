1. Instalación de DeepSparse
   Primero necesitas instalar DeepSparse. Se recomienda hacerlo en un entorno virtual de Python para evitar conflictos de dependencias.

bash
Copy code
pip install "deepsparse[server,yolo,onnxruntime]" 2. Obtener el modelo YOLOv5 en formato ONNX
DeepSparse funciona con modelos en formato ONNX. Puedes usar modelos preentrenados y optimizados que proporciona SparseZoo o convertir tu propio modelo YOLOv5 a ONNX.

Modelos de ejemplo en SparseZoo:

YOLOv5 estándar: zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/base-none
YOLOv5 podado y cuantizado (mejor rendimiento): zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned65_quant-none
Para probar con uno de estos modelos, usa uno de los stubs proporcionados.

3. Usar la API de Python para realizar inferencia
   Una vez que tengas instalado DeepSparse y el modelo en ONNX, puedes ejecutar inferencia con el siguiente código en Python.

Código de ejemplo:
python
Copy code
from deepsparse import Pipeline

# Lista de imágenes locales para inferencia

images = ["basilica.jpg"]

# Modelo YOLOv5 optimizado en SparseZoo (puedes cambiar el modelo si es necesario)

model_stub = "zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned65_quant-none"

# Crear la pipeline de DeepSparse para YOLO

yolo_pipeline = Pipeline.create(
task="yolo",
model_path=model_stub,
)

# Ejecutar inferencia en las imágenes

pipeline_outputs = yolo_pipeline(images=images, iou_thres=0.6, conf_thres=0.001)

# Imprimir los resultados

print(pipeline_outputs) 4. Ejecución de un servidor HTTP con FastAPI
Si prefieres ejecutar el modelo como un servicio web, puedes usar el servidor que viene integrado en DeepSparse.

Comando para levantar el servidor:
bash
Copy code
deepsparse.server --task yolo --model_path zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned65_quant-none
Esto creará un endpoint HTTP donde puedes enviar imágenes para inferencia.

Ejemplo de solicitud HTTP:
python
Copy code
import json
import requests

# Cargar la imagen local

path = ["basilica.jpg"]
files = [("request", open(img, "rb")) for img in path]

# Enviar solicitud al servidor para realizar inferencia

url = "http://0.0.0.0:5543/predict/from_files"
resp = requests.post(url=url, files=files)

# Obtener los resultados de las cajas delimitadoras

annotations = json.loads(resp.text)
bounding_boxes = annotations["boxes"]
labels = annotations["labels"]

print(bounding_boxes, labels) 5. Anotar imágenes con las predicciones de YOLOv5
Si quieres guardar las imágenes anotadas con las detecciones, puedes usar la CLI de DeepSparse para generar las imágenes procesadas.

Comando para anotar una imagen:
bash
Copy code
deepsparse.object_detection.annotate --model_filepath zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned65_quant-none --source basilica.jpg
Esto generará una carpeta annotation-results con la imagen anotada con las detecciones realizadas por el modelo.

6. Benchmarking de rendimiento
   Si deseas comparar el rendimiento de DeepSparse contra otros entornos de inferencia como ONNX Runtime, puedes usar el siguiente comando para ejecutar un benchmark en lotes de imágenes.

Comando para comparar con ONNX Runtime:
bash
Copy code
deepsparse.benchmark zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/base-none -s sync -b 32 -nstreams 1 -e onnxruntime
El comando comparará el rendimiento de DeepSparse frente a ONNX Runtime para un lote de 32 imágenes.

Resumen del proceso:
Instalas DeepSparse.
Obtienes un modelo YOLOv5 en formato ONNX (desde SparseZoo o convirtiéndolo).
Usas la API de Python o despliegas un servidor HTTP con FastAPI.
Opcionalmente, ejecutas benchmarks o anotas imágenes con predicciones.
