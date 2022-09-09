from openvino.inference_engine import IECore
import numpy                   as np
from   imutils    			   import paths
import cv2

def prepare_image(image, target_size=(224, 224), target_layout="NCWH"):

    # Resize image, [H, W, C] -> [300, 300, C]
    image_copy = cv2.resize(image, target_size)

    # Swap axes, [H, W, C] -> [C, H, W]
    if target_layout == "NCHW":
        image_copy = np.swapaxes(image_copy, 0, 2)
        image_copy = np.swapaxes(image_copy, 1, 2)
    
    # Expand dimensions, [1, C, H, W]
    image_copy = np.expand_dims(image_copy, 0)

    return image_copy

classes = ['dandelion','daisy','tulips','sunflowers','roses']

xml = './test_model/test_model.xml'
bin = './test_model/test_model.bin'

ie = IECore()

network = ie.read_network(xml, bin)

input_name = next(iter(network.input_info))
input_data = network.input_info[input_name].input_data
input_shape = input_data.shape # [1, 3, 300, 300]
input_layout = input_data.layout # NCHW
input_size = (input_shape[2], input_shape[3]) # (300, 300)

exec_network = ie.load_network(network=network, device_name="MYRAID", num_requests=1)

print("[INFO] sampling images...")
imagePaths = np.array(list(paths.list_images('./dataset')))
idxs = np.random.randint(0, len(imagePaths), size=(10,))
imagePaths = imagePaths[idxs]
print (imagePaths)
for image in imagePaths:
    image = cv2.imread(image)
    image_prepared = prepare_image(image, target_size=input_size, target_layout=input_layout)

    # Make inference
    output = exec_network.infer({input_name: image_prepared})

    # Get results from output layer "DetectionOutput"
    results = output
    results = np.squeeze(results['StatefulPartitionedCall/model/dense_1/Softmax']).tolist()
    prediction = classes[results.index(max(results))]
  
    cv2.putText(image, "Label: {}".format(prediction),
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
