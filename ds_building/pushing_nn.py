import os
import re
import pickle
import numpy as np
import random

import tensorflow as tf
print("TensorFlow version:", tf.__version__)

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
            y_temp.append(pickle.load(fl)) 

    
    raw = list(zip(x_temp, y_temp))
    random.shuffle(raw)
    for i in range(600,len(raw)):
        x_test.append(raw[i][0])
        if raw[i][1]:
            y_test.append(1)
        else:
            y_test.append(0)
    for i in range(600):
        x_train.append(raw[i][0])
        if raw[i][1]:
            y_train.append(1)
        else:
            y_train.append(0)

    return (np.asarray(x_train), np.asarray(y_train)), (np.asarray(x_test), np.asarray(y_test))





# get train in, train out and test in test out
(x_train, y_train), (x_test, y_test) = load_dataset()
# x_train, x_test = x_train / 255.0, x_test / 255.0



model = tf.keras.models.Sequential([
    # input layer
    tf.keras.layers.Flatten(input_shape=((126,))),
    # middle layers
    tf.keras.layers.Dense(500, activation='relu'),
    tf.keras.layers.Dense(500, activation='relu'),
    tf.keras.layers.Dense(500, activation='relu'),
    # tf.keras.layers.Dense(200, activation='sigmoid'),
    # testing purpose layer
    tf.keras.layers.Dropout(0.2),
    # output layer
    tf.keras.layers.Dense(1)
])

# model = tf.keras.models.load_model('model_200-5000-20')

# starting predictions
predictions = model(x_train[:1]).numpy()
print(predictions)

# logit to probability
# print(tf.nn.softmax(predictions).numpy())

# declare loss function
loss_fn = tf.keras.losses.MeanSquaredError()

# print loss
print(loss_fn(y_train[:1], predictions).numpy())

# compile model
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

# training
train_h = model.fit(x_train, y_train, epochs=1000, batch_size=20)

# model_hidden-layer-neurons_epochs_batch-size
model.save('model_s_500-500-500-1003-20')

print ('finished fitting')
# testing
test_h = model.evaluate(x_test,  y_test, verbose=2, batch_size = 20)

with open('train_h','wb') as f:
    pickle.dump(train_h,f) 
