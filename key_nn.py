# KeyModel neural network training

import os
import pickle
import numpy as np
import random
from graphs import plot
import tensorflow as tf
from globals import key_map

model_version = '0-84'


# Load pickle files 
# xdir: directory of x files
# ydir: directory of y files
# finger: name of finger directory
# 
# x: one hand point list (63 points) 
# y: letter corresponding to x. Will be converted to 1-hot encoded array according to available_keys 
def load_dataset(xdir, ydir, finger):
    available_keys = key_map[finger]
    x_temp = []
    y_temp = []
    for f in os.listdir('./inputs/'+xdir+'/'+finger):
        with (open('./inputs/'+xdir+'/'+finger +'/'+ f, 'rb')) as fl:
            x_temp.append(pickle.load(fl))
        
        with (open('./inputs/'+ydir+'/'+finger+'/'+f, 'rb')) as fl:
            y_temp.append(pickle.load(fl))

    
    
    raw = list(zip(x_temp, y_temp))
    random.shuffle(raw)
    
    x_final = []
    y_final = []
    
    for i in range(len(x_temp)):
        # Avoid incorrectly classified keys
        if raw[i][1] not in available_keys:
            continue

        x_final.append(raw[i][0])
        y_final.append([0 for i in range(len(available_keys))])
        y_final[-1][available_keys.index(raw[i][1])] = 1

    return (np.asarray(x_final), np.asarray(y_final))

# returns a new KeyModel
def new_model(outs):
    model = tf.keras.models.Sequential([
        # input layer
        tf.keras.layers.Flatten(input_shape=((63,))),
        
        # middle layers
        tf.keras.layers.Dense(30, activation='sigmoid'),
        
        # output layer
        tf.keras.layers.Dense(outs, activation='softmax')
    ])
    
    return model



# Models setup
models = {'sx_mig':new_model(3),'sx_anu':new_model(3),'sx_mid':new_model(3),'sx_ind':new_model(6),
          'dx_ind':new_model(6),'dx_mid':new_model(3),'dx_anu':new_model(3),'dx_mig':new_model(3)}
# models = {'dx_mid':new_model(3)}


loss_fn = tf.keras.losses.CategoricalCrossentropy()

# Custom callback to be able to plot accuracy and loss while training
class Cb(tf.keras.callbacks.Callback):
    def __init__(self,name) -> None:
        super().__init__()
        self.name = name
        self.l = {'categorical_accuracy':[],'loss':[],'val_loss':[],'val_categorical_accuracy':[]}

    def on_epoch_end(self, epoch, logs=None):
        for k in logs:
            self.l[k].append(logs[k])

        if(epoch % 50 == 0):
            plot(self.l,self.name)

        if(epoch > 300):
            self.model.stop_training = True

es = tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', mode=max, patience=300, restore_best_weights=True)


# training
for m in models:
    models[m].compile(optimizer='adam',loss=loss_fn,metrics=[tf.keras.metrics.CategoricalAccuracy()])
    
    x_train, y_train =load_dataset('x','y',m)
    x_test, y_test = load_dataset('x-test','y-test',m)
    
    models[m].fit(x_train, y_train, epochs=5000, batch_size=10, validation_data = (x_test, y_test), callbacks=[Cb(m),es])
    models[m].save('./key_models/model-'+model_version+'/' + m)

print ('Finished training')