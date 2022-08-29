import numpy                      as np
import tensorflow                 as tf
import tensorflow_datasets        as tfds
from   tensorflow.keras.callbacks import ModelCheckpoint
from   tensorflow                 import keras

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
    input_shape = (224,224,3),
    include_top = False # do not include Imagenet classifer
)

# freeze the base model
base_model.trainable = False
print("[INFO] FREEZEING LAYERS")

inputs = keras.Input(shape=(224,224,3))
x      = data_augmentation(inputs)

scale_layer = keras.layers.Rescaling(scale=1 / 127.5, offset=-1)
x           = scale_layer(x)

x       = base_model(x, training=False)
x       = keras.layers.GlobalAveragePooling2D()(x)
x       = keras.layers.Dropout(0.2)(x)  # Regularize with dropout
outputs = keras.layers.Dense(1)(x)
model   = keras.Model(inputs, outputs)

model.summary()

# put the model together
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy()],
)

model.fit(train_ds, epochs=20, validation_data=validation_ds)

base_model.trainable = True
print("[INFO] UNFREEZING LAYERS")

model.compile(
    optimizer=keras.optimizers.Adam(1e-5),  # Low learning rate to avoid over fitting
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy()],
)

# set up checkpointing
checkpoint = ModelCheckpoint('./test_model.hdf5', monitor="val_loss",save_best_only=True, verbose=1)
callbacks = [checkpoint]

model.fit(train_ds, epochs=10, validation_data=validation_ds, callbacks=callbacks)
