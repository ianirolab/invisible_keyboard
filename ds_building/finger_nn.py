import os
import re
import pickle
import numpy as np
import random
import copy
from in_model_manager import getModel0, getModel1

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
        y_test.append(raw[i][1])
    
    for i in range(int(len(raw) * trainperc)):
        x_train.append(raw[i][0])
        y_train.append(raw[i][1])

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
        y_temp[i] = raw[i][1]

    return (np.asarray(x_temp), np.asarray(y_temp))

# get train in, train out and test in test out
(x_train, y_train), (x_test, y_test) = load_dataset2()
# x_train, x_test = x_train / 255.0, x_test / 255.0



def new_model():
    model = tf.keras.models.Sequential([
        # input layer
        tf.keras.layers.Flatten(input_shape=((126,))),
        # middle layers
        tf.keras.layers.Dense(250, activation='sigmoid'),
        # tf.keras.layers.Dense(200, activation='sigmoid'),
        # testing purpose layer
        tf.keras.layers.Dropout(0.2),
        # output layer
        tf.keras.layers.Dense(10, activation='softmax')
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
loss_fn = tf.keras.losses.CategoricalCrossentropy()


# compile model
model.compile(optimizer='adam',
            loss=loss_fn,
            metrics=['accuracy'])

# training
train_h = model.fit(x_train, y_train, epochs=300, batch_size=50)

# model_hidden-layer-neurons_epochs_batch-size
model.save('model-0-72')

print ('finished fitting')
# testing
test_h = model.evaluate(x_test,  y_test, verbose=1, batch_size = 20)

x_test1,y_test1 = load_test()
model.evaluate(x_test1, y_test1, verbose=1, batch_size=10)

with open('train_h','wb') as f:
    pickle.dump(train_h,f) 

