# USAGE

# import the necessary packages
from   preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from   preprocessing.simplepreprocessor       import SimplePreprocessor
from   preprocessing.simpledatasetloader      import SimpleDatasetLoader
from   tensorflow.keras.models                import load_model
from   imutils    							  import paths
import numpy                                  as np
import argparse
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-m", "--model", required=True, help="path to pre-trained model")
args = vars(ap.parse_args())

classes = ['dandelion','daisy','tulips','sunflowers','roses']

# grab the list of images in the dataset then randomly sample
# indexes into the image paths list
print("[INFO] sampling images...")
imagePaths = np.array(list(paths.list_images(args["dataset"])))
idxs = np.random.randint(0, len(imagePaths), size=(10,))
imagePaths = imagePaths[idxs]

# initialize the image preprocessors
sp = SimplePreprocessor(224, 224)
iap = ImageToArrayPreprocessor()

# load the dataset from disk then scale the raw pixel intensities
# to the range [0, 1]
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(imagePaths)


# load the pre-trained network
print("[INFO] loading pre-trained network...")
model = load_model(args["model"])
model.summary()

# make predictions on the images
print("[INFO] predicting...")
preds = model.predict(data).argmax(axis=1)

# loop over the sample images
for (i, imagePath) in enumerate(imagePaths):
	# load the example image, draw the prediction, and display it
	# to our screen
	image = cv2.imread(imagePath)
	cv2.putText(image, "Label: {}".format(classes[preds[i]]),
		(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
	cv2.imshow("Image", image)
	cv2.waitKey(0)

print('[INFO] DONE')
