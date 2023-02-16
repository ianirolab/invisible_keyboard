# FingerModel neural network training
# TODO: should be updated to meet key_nn standards

import os
import pickle
import numpy as np
import random
from in_model_manager import getFingerModel
from graphs import plot
import tensorflow as tf


trainperc = 0.80
model_version = '0-75'

#legacy function compatible with training_set_generator
def load_dataset_legacy():
    x_temp = []
    y_temp = []
    h_temp = []
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for f in os.listdir('./ds_building/inputs/x'):
        with (open('./ds_building/inputs/x/' + f, 'rb')) as fl:
            x = pickle.load(fl)
        
        if (len(x) < 2):
            continue
        
        tmp = []
        for i in range(21):

            tmp.append(x[0].landmark[i].x)
            tmp.append(x[0].landmark[i].y)
            tmp.append(x[0].landmark[i].z)
            
            tmp.append(x[1].landmark[i].x)
            tmp.append(x[1].landmark[i].y)
            tmp.append(x[1].landmark[i].z)
        
        x_temp.append(tmp)
        
        with (open('./ds_building/inputs/y/' + f, 'rb')) as fl:
            res = pickle.load(fl)
        y_temp.append(res.copy()) 

    
    
    raw = list(zip(x_temp, y_temp))
    random.shuffle(raw)
    for i in range(int(len(raw)*trainperc), len(raw)):
        x_test.append(raw[i][0])
        y_test.append(raw[i][1])
    
    for i in range(int(len(raw) * trainperc)):
        x_train.append(raw[i][0])
        y_train.append(raw[i][1])

    return (np.asarray(x_train), np.asarray(y_train)), (np.asarray(x_test), np.asarray(y_test))


# compatible with finger_helper
def load_dataset():
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

def load_test():
    x_temp = []
    y_temp = []
    for f in os.listdir('./ds_building/inputs/x-test'):

        with (open('./ds_building/inputs/x-test/' + f, 'rb')) as fl:
            x_temp.append(pickle.load(fl))
        
        with (open('./ds_building/inputs/y-test/' + f, 'rb')) as fl:
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



(x_train, y_train), (x_test, y_test) = load_dataset()

model = new_model()

loss_fn = tf.keras.losses.CategoricalCrossentropy()

x_test1,y_test1 = load_test()

# tf.keras.metrics.P
# compile model
model.compile(optimizer='adam',
            loss=loss_fn,
            metrics=[tf.keras.metrics.CategoricalAccuracy()])



# es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)

class Cb(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs = None):
        l = {'categorical_accuracy':[],'loss':[],'val_loss':[],'val_categorical_accuracy':[],'mytest-los':[],'mytest-acc':[]}
        with open('./ds_building/ext-test-log','wb') as f:
            pickle.dump(l,f)

    def on_epoch_end(self, epoch, logs=None):
        res = model.evaluate(x_test1, y_test1,verbose=1, batch_size=10)
        
        with open('./ds_building/ext-test-log','rb') as f:
            l = pickle.load(f)
        
        logs['mytest-los'] = res[0]
        logs['mytest-acc'] = res[1]

        for k in logs:
            l[k].append(logs[k])

        with open('./ds_building/ext-test-log','wb') as f:
            pickle.dump(l,f)
        
        if(epoch % 100 == 0):
            self.model.save('./ds_building/model-temp/model-temp-'+str(epoch))
            plot(l)

        
        if(res[1] > 0.94):
            self.model.stop_training = True

cb = Cb()
es = tf.keras.callbacks.EarlyStopping(monitor='mytest-acc', min_delta = 0.03, patience=100, restore_best_weights=True,start_from_epoch =500)

# training
train_h = model.fit(x_train, y_train, epochs=5000, batch_size=50, validation_split=0.4, callbacks=[cb,es])

# model_hidden-layer-neurons_epochs_batch-size
model.save('./ds_building/model-'+model_version)

print ('Finished fitting')

# testing
test_h = model.evaluate(x_test,  y_test, verbose=1, batch_size = 20)

model.evaluate(x_test1, y_test1, verbose=1, batch_size=10)

with open('train_h','wb') as f:
    pickle.dump(train_h,f) 

