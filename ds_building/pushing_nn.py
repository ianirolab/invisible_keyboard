import os
import re
import pickle
import numpy as np
import random
import copy
from in_model_manager import getModel0, getModel1
from graphs import plot

import tensorflow as tf
print("TensorFlow version:", tf.__version__)


trainperc = 0.80

#compatible with training_set_generator
def load_dataset():
    x_temp = []
    y_temp = []
    h_temp = []
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for f in os.listdir('x'):
        with (open('./x/' + f, 'rb')) as fl:
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
        
        with (open('./y/' + f, 'rb')) as fl:
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
def load_dataset2():

    x_temp = []
    y_temp = []
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for f in os.listdir('x'):

        with (open('./x/' + f, 'rb')) as fl:
            x_temp.append(pickle.load(fl))
        
        
        
        with (open('./y/' + f, 'rb')) as fl:
            y_temp.append(pickle.load(fl))

    
    
    raw = list(zip(x_temp, y_temp))
    random.shuffle(raw)
    for i in range(int(len(raw)*trainperc), len(raw)):
        x_test.append(raw[i][0])
        if (raw[i][1][-1] == 1): 
            y_test.append([0,1])
        else:
            y_test.append([1,0])
    
    for i in range(int(len(raw) * trainperc)):
        x_train.append(raw[i][0])
        if (raw[i][1][-1] == 1): 
            y_train.append([0,1])
        else:
            y_train.append([1,0])

    return (np.asarray(x_train), np.asarray(y_train)), (np.asarray(x_test), np.asarray(y_test))

def load_test():
    x_temp = []
    y_temp = []
    for f in os.listdir('x-test'):

        with (open('./x-test/' + f, 'rb')) as fl:
            x_temp.append(pickle.load(fl))
        
        with (open('./y-test/' + f, 'rb')) as fl:
            y_temp.append(pickle.load(fl))

    
    
    raw = list(zip(x_temp, y_temp))
    random.shuffle(raw)
    for i in range(len(x_temp)):
        x_temp[i] = raw[i][0]

        if (raw[i][1][-1] == 1): 
            y_temp[i] = [0,1]
        else:
            y_temp[i] = [1,0]

    return (np.asarray(x_temp), np.asarray(y_temp))

# get train in, train out and test in test out
(x_train, y_train), (x_test, y_test) = load_dataset2()
# x_train, x_test = x_train / 255.0, x_test / 255.0



def new_model():
    model = tf.keras.models.Sequential([
        # input layer
        tf.keras.layers.Flatten(input_shape=((126,))),
        # middle layers
        # tf.keras.layers.GaussianNoise(0.01),
        tf.keras.layers.Dense(80, activation='sigmoid'),
        # testing purpose layer
        # tf.keras.layers.Dropout(0.2),
        # output layer
        tf.keras.layers.Dense(2, activation='softmax')
    ])


    # starting predictions
    # predictions = model(x_train[:1]).numpy()
    # print(predictions)

    # logit to probability
    # print(tf.nn.softmax(predictions).numpy())


    # print loss
    # print(loss_fn(y_train[:1], predictions).numpy())


    return model

model = new_model()
# declare loss function
loss_fn = tf.keras.losses.BinaryCrossentropy()

x_test1,y_test1 = load_test()

# compile model
model.compile(optimizer='adam',
            loss=loss_fn,
            metrics=['accuracy'])



# es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
# tf.keras.callbacks.Callback().
class Cb(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs = None):
        l = {'accuracy':[],'loss':[],'val_loss':[],'val_accuracy':[],'mytest-los':[],'mytest-acc':[]}
        with open('ext-test-log','wb') as f:
            pickle.dump(l,f)

    def on_epoch_end(self, epoch, logs=None):
        res = model.evaluate(x_test1, y_test1,verbose=1, batch_size=10)
        
        with open('ext-test-log','rb') as f:
            l = pickle.load(f)
        
        logs['mytest-los'] = res[0]
        logs['mytest-acc'] = res[1]

        for k in logs:
            l[k].append(logs[k])

        with open('ext-test-log','wb') as f:
            pickle.dump(l,f)
        
        if(epoch % 100 == 0):
            self.model.save('./model-temp/model-temp-'+str(epoch))
            plot(l)

        
        if(res[1] > 0.94):
            self.model.stop_training = True

cb = Cb()

# training
train_h = model.fit(x_train, y_train, epochs=5000, batch_size=50, validation_split=0.4, callbacks=cb)

# model_hidden-layer-neurons_epochs_batch-size
model.save('model-0-73')

print ('finished fitting')
# testing
test_h = model.evaluate(x_test,  y_test, verbose=1, batch_size = 20)

model.evaluate(x_test1, y_test1, verbose=1, batch_size=10)

with open('train_h','wb') as f:
    pickle.dump(train_h,f) 

