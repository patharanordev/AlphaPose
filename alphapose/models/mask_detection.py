from vit_keras import vit
from tqdm.notebook import tqdm
from tqdm import tqdm
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from vit_keras import vit
from tensorflow.keras import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPool2D, UpSampling2D, GlobalMaxPool2D, GlobalAveragePooling2D, Conv2DTranspose, concatenate
from tensorflow.keras.layers import Dense, Dropout, Activation, Reshape, Flatten, Input, AveragePooling2D
from tensorflow.keras.models import Model, load_model

devices = ["/gpu:0"]

class MaskDetection:

    def __init__(self):

        self.image_size = 384
        self.list_model = [   
        {'model':vit.vit_l16(image_size = self.image_size, activation = 'softmax', pretrained = True, include_top = False, pretrained_top = False),
            'preprocessing':vit.preprocess_inputs,
            'size':(self.image_size,self.image_size),
            'name':'VisionTransformer_l16'}
        ]
        self.model = None

    def load_model(self, model_fpath):
        self.model = self.get_model(0)
        self.model.summary()
        self.model.load_weights(model_fpath)

    def get_model(self):
        return self.model
    
    def preprocessing_batch_size(self, X, preprocessing = vit.preprocess_inputs, batch_size = 5000):
        new_X = None
        used_X = False
        size = len(X)
        for i in tqdm(range(0,size,batch_size)):
            start = i
            end = i+batch_size if i+batch_size <= size else size
            if not used_X:
                new_X = preprocessing(X[start:end])
                used_X = True
            else:
                new_X = np.append(new_X, preprocessing(X[start:end]), axis=0)
            
        new_X = np.array(new_X)
        return new_X

    def predict_batch_size(self, X, model, batch_size):
        new_X = None
        used_X = False
        size = len(X)
        for i in tqdm(range(0,size,batch_size)):
            start = i
            end = i+batch_size if i+batch_size <= size else size
            if not used_X:
                new_X = model.predict(X[start:end])
                used_X = True
            else:
                new_X = np.append(new_X, model.predict(X[start:end]), axis=0)
            
        new_X = np.array(new_X)
        return new_X

    
    def get_model(self, index, list_model=self.list_model, dropout=0.2):
        basemodel = self.list_model[index]['model']
        preprocessing = self.list_model[index]['preprocessing']
        size = self.list_model[index]['size']
        
        inputs = basemodel.input
        
        for layer in basemodel.layers:
            layer.trainable = False
        
        x = basemodel(inputs)
        x = Flatten()(x)
        x = BatchNormalization()(x)
        
        x = Dense(256)(x)
        x = Dropout(dropout)(x)
        x = Activation('gelu')(x)
        x = BatchNormalization()(x)
        
        x = Dense(128)(x)
        x = Dropout(dropout)(x)
        x = Activation('gelu')(x)
        x = BatchNormalization()(x)
        
        x = Dense(3)(x)
        x = Dropout(dropout)(x)
        x = Activation('softmax')(x)
        
        outputs = x
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model

    

# model = get_model(0)