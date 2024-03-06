from bottle import route, run, request
import base64
import os
from PIL import Image
import io
import cv2
import numpy as np

import os
import sys

# Dosya dizini
current_directory = os.path.dirname(sys.argv[0]) if getattr(sys, 'frozen', False) else os.path.dirname(__file__)

# Dosya yollari
model_config = os.path.join(current_directory, 'Sources', 'yolov4-tiny-obj.cfg')
model_weights = os.path.join(current_directory, 'Sources', 'yolov4-tiny-obj_best.weights')
class_names_file = os.path.join(current_directory, 'Sources', 'obj.names')


# Sinif adlarini yukleme
with open(class_names_file, 'r', encoding='utf-8') as f:
    class_names = f.read().strip().split('\n')

# YOLO modelini yukle
net = cv2.dnn.readNetFromDarknet(model_config, model_weights)
# docs.opencv.org/3.4/d6/d0f/group__dnn.html
output_layer_names = net.getUnconnectedOutLayersNames()

def base64_to_image(base64_string):
    image_data = base64.b64decode(base64_string)

    # Bellekten resim islenir
    image = Image.open(io.BytesIO(image_data))
    return image

def save_image(image, output_path='Sources/'):
    # Dizin bulunamazsa olusturulur
    if not os.path.exists(output_path): 
        os.makedirs(output_path)

    image_path = os.path.join(output_path, 'image.jpg')
    image.save(image_path)
    return image_path

def detect_objects(image_path, confidence_threshold=0.2 , nms_threshold=0.3):
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    # Giris resmini model icin yeniden boyutlandirma
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # YOLO ile nesne tespiti
    detections = net.forward(output_layer_names)

    boxes = []
    confidences = []
    class_ids = []

    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > confidence_threshold:
                center_x = int(obj[0] * width)
                center_y = int(obj[1] * height)
                w = int(obj[2] * width)
                h = int(obj[3] * height)

                # Dikdortgen kose koordinatlarini hesaplama
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-maximum suppression uygulama
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

    results = {}
    for i in range(len(boxes)):
        if i in indexes:
            class_name = class_names[class_ids[i]]
            confidence = confidences[i]
            
            results[class_name] = confidence
    
    return results

@route('/web', method='POST')
def web():
    data = request.json
    base64_image = data.get('image')

    # Base64 formatindaki resmi .jpg formatina donusturme
    image = base64_to_image(base64_image)

    # .jpg formatindaki resmi dosyalara kaydetme
    image_path = save_image(image)

    # YOLOv4-tiny algoritmasini kullanarak resimleri isleme
    detected_objects = detect_objects(image_path)

    telefon_confidence = detected_objects.get("Telefon", 0.0)
    priz_confidence = detected_objects.get("Priz", 0.0)
    klavye_confidence = detected_objects.get("Klavye", 0.0)

    # Cikti olusturulur.
    result_str = f"Telefon: {telefon_confidence:.2f},Priz: {priz_confidence:.2f},Klavye: {klavye_confidence:.2f}"

    return result_str

run(host='0.0.0.0', port=8080, debug=True)