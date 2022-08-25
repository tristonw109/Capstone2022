from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Dense

class FCHeadNet:
    
    def build(baseModel, classes, D):
        headmodel = baseModel.output
        headmodel = Flatten(name="flatten")(headModel)
        headmodel = Dense(D, activation="relu")(headModel)
        headModel = Dropout(0.5)(headModel)

        headModel = Desne(classes, activation="software")

        return headModel
