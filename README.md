
# Setting up yolo34 environment

1. Create new virtual environment
2. Within project folder creater  folder and name it **yolo34**

## 1. Install dependencies using pip ( **DON’T** use pip3)
> - sudo apt-get install python3-dev
> - pip install numpy cython opencv-python


## 2. Install main packages: YOLO34-python (**DON’T** use pip3)
> - pip install yolo34py-gpu

## 3. Run `*python3 webcam_demo.py*
- Create document in yolo34 folder containing the codes [here](https://github.com/madhawav/YOLO3-4-Py/blob/master/download_models.sh)
- Save the document as models.sh
- In yolo34 terminal
> - $ sh models.sh
> - 3 folders: cfg, weights and data folder should be created in yolo34 folder
- Create new script named webcam_demo containing the following codes

````
import time

import pydarknet
from pydarknet import Detector, Image
import cv2

if __name__ == "__main__":
    # Optional statement to configure preferred GPU. Available only in GPU version.
    # pydarknet.set_cuda_device(0)
    path = "../yolo34/"
    net = Detector(bytes(path + "cfg/yolov3.cfg", encoding="utf-8"), bytes(path + "weights/yolov3.weights", encoding="utf-8"), 0,
                   bytes(path + "cfg/coco.data", encoding="utf-8"))

    cap = cv2.VideoCapture(0)

    while True:
        r, frame = cap.read()
        if r:
            start_time = time.time()

            # Only measure the time taken by YOLO and API Call overhead

            dark_frame = Image(frame)
            results = net.detect(dark_frame)
            del dark_frame

            end_time = time.time()
            # Frames per second can be calculated as 1 frame divided by time required to process 1 frame
            fps = 1 / (end_time - start_time)

            print("FPS: ", fps)
            print("Elapsed Time:", end_time - start_time)

            for cat, score, bounds in results:
                x, y, w, h = bounds
                cv2.rectangle(frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (255, 0, 0))
                cv2.putText(frame, str(cat.decode("utf-8")), (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX, 1,
                            (255, 255, 0))

            cv2.imshow("preview", frame)

        k = cv2.waitKey(1)
        if k == 0xFF & ord("q"):
            break
````
## Errors

### 1. Couldn't open file: data/coco.names
> - open **coco.data** folder in **cfg** folder and change **names = ../data/coco.names** to **names = ../yolo34/data/coco.names**
