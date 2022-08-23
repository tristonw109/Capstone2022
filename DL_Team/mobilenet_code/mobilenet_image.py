from openvino.inference_engine import IECore
import numpy as np
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
        class_ = classes[int(detection[1])]

        if class_ == 'person':
            print('PERSON IS THERE')
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
        if 1 in item:
            return True
    return False

xml = './model/ssd_mobilenet_v2_coco.xml'
bin = './model/ssd_mobilenet_v2_coco.bin'
txt = './classLabels.txt'


with open(txt,'r') as file:
    labels = file.read().splitlines()

labels = dict(zip(range(1,len(labels)),labels))

ie = IECore()

network = ie.read_network(xml, bin)


input_name = next(iter(network.input_info))
input_data = network.input_info[input_name].input_data
input_shape = input_data.shape # [1, 3, 300, 300]
input_layout = input_data.layout # NCHW
input_size = (input_shape[2], input_shape[3]) # (300, 300)


exec_network = ie.load_network(network=network, device_name="MYRIAD", num_requests=1)

image = cv2.imread('inference/photo/test.jpg')

image_prepared = prepare_image(image, target_size=input_size, target_layout=input_layout)

# Make inference
output = exec_network.infer({input_name: image_prepared})

# Get detections from output layer "DetectionOutput"
detections = output["DetectionOutput"]
print(np.squeeze(detections[0,0]))

print(isStop(detections[0,0]))
# Draw bounding boxes
image_boxed = draw_bounding_boxes(image, detections, labels)

cv2.imwrite('./output.jpg',image_boxed)
