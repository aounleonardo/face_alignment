import os
import cv2
import torch
import numpy as np
from skimage.data import astronaut

from ibug.face_alignment import FANPredictor
from ibug.face_detection import RetinaFacePredictor

image = astronaut()

device = "cpu"
if torch.cuda.is_available():
    device = torch.cuda.current_device()

face_detector = RetinaFacePredictor(
    device=device, 
    model=RetinaFacePredictor.get_model("mobilenet0.25"),
)
detections = face_detector(image, rgb=False)
assert len(detections) == 1, "Please submit an image with exactly one clear frontal face"
detection = detections[0, :4]

config = FANPredictor.create_config(gamma = 1.0, radius = 0.1, use_jit = False)
fan = FANPredictor(device=device, model=FANPredictor.get_model('2dfan2_alt'), config=config)

landmarks, scores = fan(image, detection)
