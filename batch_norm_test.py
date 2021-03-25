import os
import pickle
import cv2
import numpy as np
import torch

from ibug.face_alignment import FANPredictor
from ibug.face_alignment.utils import plot_landmarks

def main():

    with open(os.path.join("batch_norm_test_resources", torch.__version__ + ".torchconfig"), "w") as file:
        print(*torch.__config__.show().split("\n"), sep="\n", file=file)

    frame = cv2.imread(os.path.join("batch_norm_test_resources", "test_input.jpg"))
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detections = np.array(
        [[354.8009    , 207.38774   , 463.7139    , 372.86032]],
        dtype=np.float32,
    )

    open(os.path.join("batch_norm_test_resources", torch.__version__ + "_log.out"), "w").close()

    with torch.no_grad():
        model = FANPredictor.get_model("2dfan2")
        landmark_detector = FANPredictor(device="cuda:0", model=model)
        landmarks, scores = landmark_detector(image, detections, rgb=True)

    for face, lm, sc in zip(detections, landmarks, scores):
        bbox = face[:4].astype(int)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(0, 0, 255), thickness=2)
        plot_landmarks(frame, lm, sc, threshold=0.0)
        if len(face) > 5:
            plot_landmarks(frame, face[5:].reshape((-1, 2)), pts_radius=3)

    cv2.imwrite(os.path.join("batch_norm_test_resources", torch.__version__ + ".jpg"), frame)


    print(scores)

if __name__ == "__main__":
    main()
