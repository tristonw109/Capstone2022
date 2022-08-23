from openvino.inference_engine import IECore
from time import time
import RPi.GPIO as GPIO
import numpy as np
import argparse
import cv2

def prepare_image(image, target_size=(300, 300), target_layout="NCWH"):

    # Resize image, [H, W, C] -> [300, 300, C]
    image_copy = cv2.resize(image, target_size)

    # Swap axes, [H, W, C] -> [C, H, W]
    if target_layout == "NCHW":
        image_copy = np.swapaxes(image_copy, 0, 2)
        image_copy = np.swapaxes(image_copy, 1, 2)
    
    # Expand dimensions, [1, C, H, W]
    image_copy = np.expand_dims(image_copy, 0)

    return image_copy
def draw_bounding_boxes(image, detections, classes, threshold=0.5, box_color=(255, 0, 0)):

    image_copy = np.copy(image)

    # Get image dimensions
    image_height = image_copy.shape[0]
    image_width = image_copy.shape[1]

    # Iterate through detections
    num_detections = detections.shape[2]
    
    for i in range(num_detections):

        detection = detections[0, 0, i]

        # Skip detections with confidence below threshold
        confidence = detection[2]
        if confidence < threshold:
            continue

        # Draw bounding box
        x_min = int(detection[3]*image_width)
        y_min = int(detection[4]*image_height)

        x_max = int(detection[5]*image_width)
        y_max = int(detection[6]*image_height)

        top_left = (x_min, y_min)
        bottom_right = (x_max, y_max)

        cv2.rectangle(image_copy, top_left, bottom_right, box_color, 2)
        # Get class text
        if int(detection[1]) in range(len(classes)):
            class_ = classes[int(detection[1])]
        else:
            class_ = None

        # Draw text background
        text_size = cv2.getTextSize(class_, cv2.FONT_HERSHEY_PLAIN, 2, 1)
        
        top_left = (x_min, y_max-text_size[0][1])
        bottom_right = (x_min+text_size[0][0], y_max)

        cv2.rectangle(image_copy, top_left, bottom_right, box_color, cv2.FILLED)

        # Draw text
        cv2.putText(image_copy, class_, (x_min,y_max), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1)

    return image_copy
def isStop(lst) -> bool:
    for item in lst:
        if 13 in item:
            return True
    return False

ap = argparse.ArgumentParser()
ap.add_argument("-x", "--xml",required=True,help="path to .xml model")
ap.add_argument("-b", "--bin",required=True,help="path to .bin weights")
ap.add_argument("-l", "--labels",required=True,help="path to .txt file of class labels")
ap.add_argument("-v", "--video",help="path to the (optional) video file")
args = vars(ap.parse_args())

GPIO.cleanup()
GPIO.setmode(GPIO.BOARD) 
GPIO.setup(3,GPIO.OUT)
GPIO.output(3,GPIO.LOW)


xml = args['xml']
bin = args['bin']

avgStop = []

# if a video path was not supplied, grab the reference to the webcam
if not args.get("video", False):
	camera = cv2.VideoCapture(0)

# otherwise, load the video
else:
	camera = cv2.VideoCapture(args["video"])



with open(args['labels'],'r') as file:
    labels = file.read().splitlines()

labels = dict(zip(range(1,len(labels)),labels))

ie = IECore() # init the inference core

network = ie.read_network(xml, bin) #set up network with models

# collect network infomation: 
input_name = next(iter(network.input_info))
input_data = network.input_info[input_name].input_data
input_shape = input_data.shape 
input_layout = input_data.layout 
input_size = (input_shape[2], input_shape[3]) 



device = 'MYRIAD'
infer_times = [] 

exec_network = ie.load_network(network=network, device_name=device, num_requests=1)

while True:
    
    (grabbed, frame) = camera.read() # grab camera info

    if args.get("video") and not grabbed: # break if nothing is grabbed
        break

    frameClone = frame.copy() # copy frame

    frame_prepared = prepare_image(frameClone, target_size=input_size, target_layout=input_layout) #alter frame to match model info

    start = time()
    output = exec_network.infer({input_name: frame_prepared}) # make inference on prepared fram
    stop = time()

    infer_times.append(stop-start)
    
    detections = output["DetectionOutput"] # Get detections from output layer "DetectionOutput"

    print(isStop(detections[0,0]))
    GPIO.output(3,GPIO.HIGH) if isStop(detections[0,0]) else  GPIO.output(3,GPIO.LOW) 

    image_boxed = draw_bounding_boxes(frame, detections, labels) # Draw bounding boxes

    #cv2.imshow('output',image_boxed) # show image

    if cv2.waitKey(1) & 0xFF == ord("q"): # if you press q, stop
        break

camera.release() # close camera
cv2.destroyAllWindows() # destory(close) all cv2 opened windows
GPIO.cleanup() # CLean-up/Destory any pin profiles that were set-up

avg_infer = sum(infer_times)/len(infer_times)
print('Average time taking to inference: {:.5} ms'.format(avg_infer))
