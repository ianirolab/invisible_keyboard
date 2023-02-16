import os
import re
import pickle
import numpy as np
import random
import copy
from ds_building.in_model_manager import getModel0, getModel1
from graphs import plot

import tensorflow as tf
print("TensorFlow version:", tf.__version__)

key_map = {'sx_mig':['q','a','z'],'sx_anu':['w','s','x'],'sx_mid':['e','d','c'],'sx_ind':['r','t','f','g','c','v'],
          'dx_ind':['y','u','h','j','n','m'],'dx_mid':['i','k',','],'dx_anu':['o','l','.'],'dx_mig':['p',';','/']}


# compatible with finger_helper
def load_dataset(path, arr):

    x_temp = []
    y_temp = []
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for f in os.listdir('./x2/'+path):

        with (open('./x2/'+path +'/'+ f, 'rb')) as fl:
            x_temp.append(pickle.load(fl))
        
        
        
        with (open('./y2/'+path+'/'+f, 'rb')) as fl:
            y_temp.append(pickle.load(fl))

    
    
    raw = list(zip(x_temp, y_temp))
    random.shuffle(raw)
    x_final = []
    y_final = []
    
    for i in range(len(x_temp)):
        if raw[i][1] not in arr:
            continue
        x_final.append(raw[i][0])
        y_final.append([0 for i in range(len(arr))])
        y_final[-1][arr.index(raw[i][1])] = 1

    return (np.asarray(x_final), np.asarray(y_final))


def load_test(path,arr):
    x_temp = []
    y_temp = []
    for f in os.listdir('./x-test2/'+path):

        with (open('./x-test2/'+path +'/'+ f, 'rb')) as fl:
            x_temp.append(pickle.load(fl))
        
        
        
        with (open('./y-test2/'+path+'/'+f, 'rb')) as fl:
            y_temp.append(pickle.load(fl))

    
    
    raw = list(zip(x_temp, y_temp))
    random.shuffle(raw)
    x_final = []
    y_final = []
    
    for i in range(len(x_temp)):
        if raw[i][1] not in arr:
            continue
        x_final.append(raw[i][0])
        y_final.append([0 for i in range(len(arr))])
        y_final[-1][arr.index(raw[i][1])] = 1

    return (np.asarray(x_final), np.asarray(y_final))

# get train in, train out and test in test out
# x_train, x_test = x_train / 255.0, x_test / 255.0



def new_model(outs):
    model = tf.keras.models.Sequential([
        # input layer
        tf.keras.layers.Flatten(input_shape=((63,))),
        # middle layers
        # tf.keras.layers.GaussianNoise(0.005),
        tf.keras.layers.Dense(30, activation='sigmoid'),
        # testing purpose layer
        # tf.keras.layers.Dropout(0.2),
        # output layer
        tf.keras.layers.Dense(outs, activation='softmax')
    ])
    

    return model

# models = {'sx_mig':new_model(3),'sx_anu':new_model(3),'sx_mid':new_model(3),'sx_ind':new_model(6),
#           'dx_ind':new_model(6),'dx_mid':new_model(3),'dx_anu':new_model(3),'dx_mig':new_model(3)}
models = {'sx_ind':new_model(6),
          'dx_ind':new_model(6),}

# declare loss function
loss_fn = tf.keras.losses.CategoricalCrossentropy()





# es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
# tf.keras.callbacks.Callback().
class Cb(tf.keras.callbacks.Callback):
    def __init__(self,name) -> None:
        super().__init__()
        self.name = name
        self.l = {'categorical_accuracy':[],'loss':[],'val_loss':[],'val_categorical_accuracy':[]}

    def on_epoch_end(self, epoch, logs=None):
        # res = self.model.evaluate(x_test1, y_test1,verbose=1, batch_size=10)
        
        # logs['mytest-los'] = res[0]
        # logs['mytest-acc'] = res[1]

        for k in logs:
            self.l[k].append(logs[k])

        
        if(epoch % 50 == 0):
            plot(self.l,self.name)

        
        if(epoch > 300):
            self.model.stop_training = True

es = tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', mode=max, patience=300, restore_best_weights=True)


# training
for m in models:
    print(m)
    cb = Cb(m)
    models[m].compile(optimizer='adam',
            loss=loss_fn,
            metrics=[tf.keras.metrics.CategoricalAccuracy()])
    x_test1, y_test1 = load_test(m, key_map[m])
    
    x_train, y_train =load_dataset(m,key_map[m])
    models[m].fit(x_train, y_train, epochs=5000, batch_size=10, validation_data = (x_test1, y_test1), callbacks=[cb,es])
    models[m].save('./key_models/model-0-83/' + m)

# model_hidden-layer-neurons_epochs_batch-size

print ('finished fitting')
# testing
# test_h = model.evaluate(x_test,  y_test, verbose=1, batch_size = 20)


