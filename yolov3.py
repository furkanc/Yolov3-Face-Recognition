import cv2
import argparse
import os
import numpy as np
from face_recog import FaceRecog
import random

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", required=True, help="Path to input video")
parser.add_argument("-o", "--output", required=True, help="Path to output video")
parser.add_argument("-y", "--yolo", required=True, help="Path to YOLO")
parser.add_argument("-c", "--confidence", type=float, default=0.4, help="Confidence threshold")
parser.add_argument("-n", "--nms", type=float, default=0.5, help="Non-maximum suppression threshold")
parser.add_argument("-f", "--face_embeddings", type=bool, default=True, help="Prepare a new Face embeddings file or not")
args = parser.parse_args()

# Load class names
print("Loading Classes..")
CLASSES = None
class_path = os.path.sep.join([args.yolo, "coco.names"])
with open(class_path, 'rt') as f:
    CLASSES = f.read().rstrip('\n').split('\n')
print("Done..")

# Colors for bounding boxes
np.random.seed(12)
COLORS = np.random.randint(0, 255, size=(len(CLASSES), 3), dtype="uint8")


# Load Face Recognize Module
print("Building Face Recognition Module..")
face_recognizer = FaceRecog()
face_recognizer.build(0.1, args.face_embeddings)
print("Done..")

# Load Network
print("Loading YOLOv3 weights...")
weight_path = os.path.sep.join([args.yolo, "yolov3.weights"])
config_path = os.path.sep.join([args.yolo, "yolov3.cfg"])
net = cv2.dnn.readNetFromDarknet(config_path, weight_path)
print("Done...")

# get only output layer names from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

cap = cv2.VideoCapture(args.input)

fps = cap.get(cv2.CAP_PROP_FPS)
writer = cv2.VideoWriter('{}.avi'.format(args.output),
                         cv2.VideoWriter_fourcc(*'XVID'), int(fps), (1280, 720))
(W, H) = (None, None)

print("Detection has started.It may take a while...")
while True:
    hasFrame, frame = cap.read()

    # if there is no frame, we've reached the end.
    if not hasFrame:
        cv2.waitKey()
        break
    # get frame dimensions for the first time
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # create a blob from frame and forward it to YOLO.
    # it will give us the bounding boxes and probabilities
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(ln)

    # initialize detected boxes, confidences, class IDs
    boxes = []
    confs = []
    class_IDs = []

    for output in outputs:

        for detected in output:

            # get class ID and probability.
            scores = detected[5:]
            class_ID = np.argmax(scores)
            confidence = scores[class_ID]

            # check if probability higher than confidence
            if confidence > args.confidence:
                # scaling the bounding boxes respect to frame size
                # YOLO returns the bounding box parameters in the order of :
                # center X , center Y, width , height
                box = detected[0:4] * np.array([W, H, W, H])
                (center_x, center_y, width, height) = box.astype("int")

                # find top and left corner of bounding box
                x = int(center_x - (width / 2))
                y = int(center_y - (height / 2))

                # add to list
                boxes.append([x, y, int(width), int(height)])
                confs.append(float(confidence))
                class_IDs.append(class_ID)

    # applying non-maxima suppression
    idxs = cv2.dnn.NMSBoxes(boxes, confs, args.confidence, args.nms)

    if len(idxs) > 1:
        # get coordinates of bounding box
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            color = [int(c) for c in COLORS[class_IDs[i]]]
            text_color = color.copy()
            # draw the bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 5)
            class_id = CLASSES[class_IDs[i]]

            # check if person identified
            # if the person identified , give the label name as its name.
            if class_id == 'person':
                name, text_color = face_recognizer.find_face(frame, x, y, w, h)
                text = "{}: {:.4f}".format(name, confs[i])
            else:
                text = "{}: {:.4f}".format(CLASSES[class_IDs[i]], confs[i])

            cv2.putText(frame, text, (x, y - 5), cv2.FONT_ITALIC, 0.9, text_color, 2)

    writer.write(frame)
print("Program succesfully ended...")
writer.release()
cap.release()
