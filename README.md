# ibug.face_alignment
2D facial landmark detector based on [FAN](http://openaccess.thecvf.com/content_ICCV_2017/papers/Bulat_How_Far_Are_ICCV_2017_paper.pdf) \[1\] with some pretrained weights. Our training code is available in this repostory: [https://github.com/hhj1897/fan_training](https://github.com/hhj1897/fan_training).

## Prerequisites
* [Numpy](https://www.numpy.org/): `$pip3 install numpy`
* [OpenCV](https://opencv.org/): `$pip3 install opencv-python`
* [PyTorch](https://pytorch.org/): `$pip3 install torch torchvision`
* [ibug.face_detection](https://github.com/hhj1897/face_detection) (only needed by the test script): See this repository for details: [https://github.com/hhj1897/face_detection](https://github.com/hhj1897/face_detection).

## How to Install
```
git clone https://github.com/hhj1897/face_alignment.git
cd face_alignment
pip install -e .
```

## How to Test
* To test on live video: `python face_alignment_test.py [-i webcam_index]`
* To test on a video file: `python face_alignment_test.py [-i input_file] [-o output_file]`

### Docker
You can run a containerized demo using Docker. It will install the needed dependencies and allow you to test the FANPredictor on a sample image.

Either run `bash demo/run.sh` or `docker build -t ibug-face_alignment -f ./demo/Dockerfile . && docker run -it --rm ibug-face_alignment`
You will automatically enter a tmux session and run demo.py script. If you don't want this, you can either kill it immediately or provide "--entrypoint /bin/bash" to the docker run command.

Please install Docker using [the official instructions](https://docs.docker.com/get-docker/)

**NOTE** If you're running on Windows, running the container might fail, and you would need to replace `docker run` with `winpty docker run`. I also recommend running the commands from a Unix-shell, for example Git Bash, or Docker Quickstart Terminal. This is taken care of if you use the shell script.

## How to Use
```python
# Import the libraries
import cv2
from ibug.face_detection import RetinaFacePredictor
from ibug.face_alignment import FANPredictor
from ibug.face_alignment.utils import plot_landmarks

# Create a RetinaFace detector using Resnet50 backbone, with the confidence 
# threshold set to 0.8
face_detector = RetinaFacePredictor(
    thershold=0.8, device='cuda:0',
    model=RetinaFacePredictor.get_model('resnet50'))

# Create a facial landmark detector
landmark_detector = FANPredictor(
    device='cuda:0', model=FANPredictor.get_model('2dfan2_alt'))

# Load a test image. Note that images loaded by OpenCV adopt the B-G-R channel
# order.
image = cv2.imread('test.png')

# Detect faces from the image
detected_faces = face_detector(image, rgb=False)

# Detect landmarks from the faces
# Note: 
#   1. The input image must be a byte array of dimension HxWx3.
#   2. The input face boxes must be a array of dimension Nx4, N being the 
#      number of faces. More columns are allowed, but only the first 4
#      columns will be used (which should be the left, top, right, and
#      bottom coordinates of the face).
#   3. The returned landmarks are stored in a Nx68x2 arrays, each row giving
#      the X and Y coordinates of a landmark.
#   4. The returned scores are stored in a Nx68 array. Scores are usually
#      within the range of 0 to 1, but could go slightly beyond.
landmarks, scores = landmark_detector(image, detected_faces, rgb=False)

# Draw the landmarks onto the image
for lmks, scs in zip(landmarks, scores):
    plot_landmarks(image, lmks, scs, threshold=0.2)
```

## References
\[1\] Bulat, Adrian, and Georgios Tzimiropoulos. "[How far are we from solving the 2d & 3d face alignment problem?(and a dataset of 230,000 3d facial landmarks).](http://openaccess.thecvf.com/content_ICCV_2017/papers/Bulat_How_Far_Are_ICCV_2017_paper.pdf)" In _Proceedings of the IEEE International Conference on Computer Vision_, pp. 1021-1030. 2017.
