# FingerModel neural network training
# TODO: should be updated to meet key_nn standards

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pickle
import numpy as np
import random
from in_model_manager import getFingerModel
from graphs import plot
import tensorflow as tf


trainperc = 0.80
model_version = '0-75'


# compatible with finger_helper
def load_dataset_fh():
    x_temp = []
    y_temp = []
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for f in os.listdir('x'):

        with (open('./ds_building/inputs/x/' + f, 'rb')) as fl:
            x_temp.append(pickle.load(fl))
        
        
        
        with (open('./ds_building/inputs/y/' + f, 'rb')) as fl:
            y_temp.append(pickle.load(fl))

    
    
    raw = list(zip(x_temp, y_temp))
    random.shuffle(raw)
    for i in range(int(len(raw)*trainperc), len(raw)):
        x_test.append(raw[i][0])
        y_test.append(raw[i][1])
    
    for i in range(int(len(raw) * trainperc)):
        x_train.append(raw[i][0])
        y_train.append(raw[i][1])

    return (np.asarray(x_train), np.asarray(y_train)), (np.asarray(x_test), np.asarray(y_test))

def load_dataset(test = False):
    x_dir = 'x-test' if test else 'x' 
    y_dir = 'y-test' if test else 'y' 
    x_temp = []
    y_temp = []
    for f in os.listdir('./ds_building/inputs/'+x_dir):

        with (open('./ds_building/inputs/'+x_dir+'/' + f, 'rb')) as fl:
            x_temp.append(pickle.load(fl))
        
        with (open('./ds_building/inputs/'+y_dir+'/' + f, 'rb')) as fl:
            y_temp.append(pickle.load(fl))

    
    
    raw = list(zip(x_temp, y_temp))
    random.shuffle(raw)
    for i in range(len(x_temp)):
        x_temp[i] = raw[i][0]
        y_temp[i] = raw[i][1]

    return (np.asarray(x_temp), np.asarray(y_temp))

def new_model():
    model = tf.keras.models.Sequential([
        # input layer
        tf.keras.layers.Flatten(input_shape=((126,))),
        # middle layers
        tf.keras.layers.GaussianNoise(0.01),
        tf.keras.layers.Dense(60, activation='sigmoid'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    return model



x_train, y_train = load_dataset()
x_test,  y_test  = load_dataset(test=True)

model = new_model()

loss_fn = tf.keras.losses.CategoricalCrossentropy()


# tf.keras.metrics.P
# compile model
model.compile(optimizer='adam',
            loss=loss_fn,
            metrics=[tf.keras.metrics.CategoricalAccuracy()])



# es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)

class Cb(tf.keras.callbacks.Callback):
    def __init__(self) -> None:
        super().__init__()
        self.name = 'finger'
        self.l = {'categorical_accuracy':[],'loss':[],'val_loss':[],'val_categorical_accuracy':[]}

    def on_epoch_end(self, epoch, logs=None):
        for k in logs:
            self.l[k].append(logs[k])

        if(epoch % 50 == 0):
            plot(self.l,self.name)

        if(epoch > 300):
            self.model.stop_training = True

cb = Cb()
es = tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', mode=max, patience=300, restore_best_weights=True)

# training
train_h = model.fit(x_train, y_train, epochs=5000, batch_size=50, validation_data=(x_test,y_test), callbacks=[cb,es])

# model_hidden-layer-neurons_epochs_batch-size
model.save('./ds_building/model-'+model_version)

print ('Finished fitting')


