from time import sleep
import cv2 as cv
import numpy as np
import os
from glob import glob

# Initialize the parameters
confThreshold = 0.5  # Confidence threshold
nmsThreshold = 0.4   # Non-maximum suppression threshold
inpWidth = 416       # Width of network's input image
inpHeight = 416      # Height of network's input image

# Load class names
classesFile = "obj.names"
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Load the model configuration and weights
modelConfiguration = "yolov3-obj.cfg"
modelWeights = "yolov3-obj_2400.weights"
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# Get the names of the output layers
def getOutputsNames(net):
    layersNames = net.getLayerNames()
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Draw the predicted bounding box
def drawPred(classId, conf, left, top, right, bottom):
    global frame_count
    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
    label = f'{classes[classId]}: {conf:.2f}'
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), 
                 (left + round(1.5 * labelSize[0]), top + baseLine), (255, 255, 255), cv.FILLED)
    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)

# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs):
    frameHeight, frameWidth = frame.shape[:2]
    boxes, confidences, classIds = [], [], []

    # Scan through all the bounding boxes
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x, center_y, width, height = (detection[0:4] * [frameWidth, frameHeight, frameWidth, frameHeight]).astype('int')
                left, top = int(center_x - width / 2), int(center_y - height / 2)
                boxes.append([left, top, width, height])
                confidences.append(float(confidence))
                classIds.append(classId)

    # Perform non-maximum suppression
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    helmet_detected = False

    for i in indices:
        i = i[0]
        box = boxes[i]
        left, top, width, height = box
        drawPred(classIds[i], confidences[i], left, top, left + width, top + height)
        if classes[classIds[i]] == 'helmet':  # Assuming 'helmet' is a class in obj.names
            helmet_detected = True

    # Save frame if helmet is detected
    if helmet_detected:
        output_path = 'output_images'
        os.makedirs(output_path, exist_ok=True)
        frame_name = os.path.basename(fn)
        output_file = os.path.join(output_path, frame_name)
        cv.imwrite(output_file, frame)
        print(f"Saved: {output_file}")

# Process input images
winName = 'Helmet Detection'
cv.namedWindow(winName, cv.WINDOW_NORMAL)

# Loop through test images
for fn in glob('test_images/*.jpg'):
    frame = cv.imread(fn)
    if frame is None:
        print(f"Could not read {fn}")
        continue

    # Create a 4D blob from a frame
    blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
    net.setInput(blob)

    # Run the forward pass
    outs = net.forward(getOutputsNames(net))

    # Process the detections
    postprocess(frame, outs)

    # Show the frame
    cv.imshow(winName, frame)
    if cv.waitKey(500) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cv.destroyAllWindows()
