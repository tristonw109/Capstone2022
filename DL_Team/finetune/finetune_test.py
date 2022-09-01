import numpy                      as np
import tensorflow                 as tf
import tensorflow_datasets        as tfds
from   sklearn.preprocessing      import LabelBinarizer
from   tensorflow.keras.callbacks import ModelCheckpoint
from   keras.optimizers           import RMSprop
from   keras.optimizers           import SGD
from   newoutput_layer            import OutputLayers
from   keras.layers               import Input
from   tensorflow                 import keras
from   keras.models               import Model


print("LOADING DATASET")
((trainx,trainy),(testx,testy)) = tfds.as_numpy(tfds.load(
    "tf_flowers",
    split=["train[:40]", "train[40%:60%]"],
    batch_size=-1,
    as_supervised=True
))

trainx = np.resize(trainx, (240,240))
trainy = np.resize(trainy, (240,240))
testx = np.resize(testx, (240,240))
testy = np.resize(testy, (240,240))

print(trainx.shape)
print("Here")
trainx = trainx.astype("float") / 255.0
testx = testx.astype("float") / 255.0
print("Here1")
#lb = LabelBinarizer()
#trainy = lb.fit_transform(trainy)
#testy = lb.transform(testy)
print("Here2")
data_augmentation = keras.Sequential(
    [keras.layers.RandomFlip("horizontal"), keras.layers.RandomRotation(0.1),]
)
print("Here3")
base_model = keras.applications.VGG16(
    weights     = 'imagenet',
    input_tensor = Input(shape=(240,240,3)),
    include_top = False # do not include Imagenet classifer
)
print("Here4")
# freeze the base model
head = OutputLayers.build(D=256, baseModel=base_model, numClasses=5)

model = Model(inputs=base_model.input, outputs=head)
model.summary()

print ('[INFO] FREEZING BASE MODEL LAYERS')
for layer in base_model.layers[15:]:
    layer.trainable = False

#  put the model together
model.compile(loss="categorical_crossentropy", optimizer=RMSprop(lr=0.001), metrics=["accuracy"])
print("FIT_1")
model.fit(trainx,trainy, validation_data=(testx,testy), epochs=20, batch_size=20, verbose=1)
print("Done FIT_1")
predictions = model.predict(testy)
print (predictions)


print("[INFO] UNFREEZING LAYERS")
for layer in base_model.layers:
    layer.trainable = True

model.compile(loss="categorical_crossentropy", optimizer=SGD(lr=0.001), metrics=["accuracy"])

# set up checkpointing
checkpoint = ModelCheckpoint('./test_model.hdf5', monitor="val_loss",save_best_only=True, verbose=1)
callbacks = [checkpoint]

model.fit(trainx,trainy, validation_data=(testx,testy), epoche=100, batch_size=10, verbose=1)
