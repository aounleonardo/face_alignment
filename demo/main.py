import os
import cv2
import torch
from ibug.face_alignment import FANPredictor
from skimage.data import astronaut
import numpy as np

image = astronaut()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
face_cascade = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))
detections = face_cascade.detectMultiScale(gray)
assert len(detections) == 1, "Please submit an image with exactly one clear frontal face"
x, y, w, h = detections[0]
detection = np.array([x, y, x+w, y+h])



config = FANPredictor.create_config(gamma = 1.0, radius = 0.1, use_jit = False)

device = "cpu"
if torch.cuda.is_available():
    device = torch.cuda.current_device()
fan = FANPredictor(device=device, model=FANPredictor.get_model('2dfan2_alt'), config=config)

landmarks, scores = fan(image, detection)
