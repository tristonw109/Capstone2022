from keras.applications import VGG16
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--include-top", type=int, default=1, help="whether or not to include top of CNN")
args = vars(ap.parse_args())

print("[INFO] loading network")
model = VGG16(weights="imagenet", include_top=True)
print("[INFO] showing layers")

for (i, layers) in enumerate(model.layers):
	print ("[INFO] {}\t{}".format(i, layers.__class__.__name__))
