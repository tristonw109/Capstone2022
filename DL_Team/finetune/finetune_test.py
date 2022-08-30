import numpy                      as np
import tensorflow                 as tf
import tensorflow_datasets        as tfds
from   tensorflow.keras.callbacks import ModelCheckpoint
from   keras.optimizers           import RMSprop
from   keras.optimizers           import SGD
from   newoutput_layer            import OutputLayers
from   keras.layers               import Input
from   tensorflow                 import keras
from   keras.models               import Model

(train_ds, validation_ds, test_ds) = tfds.load(
    "cats_vs_dogs",
    split=["train[:40%]", "train[40%:50%]","train[50%:60%]"],
    as_supervised=True
)

size = (224,224)

train_ds      = train_ds.map(lambda x, y: (tf.image.resize(x, size), y))
validation_ds = validation_ds.map(lambda x, y: (tf.image.resize(x, size), y))
test_ds      = test_ds.map(lambda x, y: (tf.image.resize(x, size), y))

batch_size = 32

train_ds      = train_ds.cache().batch(batch_size).prefetch(buffer_size=10)
validation_ds = validation_ds.cache().batch(batch_size).prefetch(buffer_size=10)
test_ds       = test_ds.cache().batch(batch_size).prefetch(buffer_size=10)

data_augmentation = keras.Sequential(
    [keras.layers.RandomFlip("horizontal"), keras.layers.RandomRotation(0.1),]
)

base_model = keras.applications.VGG16(
    weights     = 'imagenet',
    input_tensor = Input(shape=(224,224,3))
    include_top = False # do not include Imagenet classifer
)

# freeze the base model
head = OutputLayers.build(D=256, baseModel=base_model, numClasses=2)

model = Model(inputs=base_model.input, outputs=head)
model.summary()

print ('[INFO] FREEZING BASE MODEL LAYERS')
for layer in base_model.layers:
    layer.trainable = False

#  put the model together
model.compile(loss="categorical_crossentropy", optimizer=RMSprop(lr=0.001), metrics=["accuracy"])

model.fit(train_ds,2, epochs=20, validation_data=validation_ds)

predictions = model.predict(test_ds)
print (predictions)


print("[INFO] UNFREEZING LAYERS")
for layer in base_model.layers:
    layer.trainable = True

model.compile(loss="categorical_crossentropy", optimizer=SGD(lr=0.001), metrics=["accuracy"])

# set up checkpointing
checkpoint = ModelCheckpoint('./test_model.hdf5', monitor="val_loss",save_best_only=True, verbose=1)
callbacks = [checkpoint]

model.fit(train_ds,2, epochs=100, validation_data=validation_ds, callbacks=callbacks)
