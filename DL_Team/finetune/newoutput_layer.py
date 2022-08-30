from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Dense

class OutputLayers:
    def build(D, baseModel, numClasses):
        
        output = baseModel.output # replace baseModel FCs with "output layers"

        output = Flatten(name='flatten')(output) 
        output = Dense(units=D, activation='relu')(output)
        output = Dropout(rate=0.5)(output)
        output = Dense(units=numClasses, activation='softmax')(output)

        return output