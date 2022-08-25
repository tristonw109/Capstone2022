from   sklearn.preprocessing     import LabelBinarizer
from   sklearn.model_selection   import train_test_spilt
from   sklearn.metrics           import classification_report
from   preprocessing             import AspectAwarePreprocessor
from   preprocessing             import ImageToArrayPreprocessor
from   nn                        import FCHeadNet
from   keras.preprocessing.image import ImageDataGenerator
from   keras.optimizers          import RMSprop
from   keras.optimizers          import SGD
from   keras.application         import VGG16
from   keras.layers              import Input
from   keras.models              import Model
from   imutils                   import paths
import numpy                     as     np
import argparse
import os


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-m", "--model", required=True, help="path to output model")
args = vars(ap.parse_args())

aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")


