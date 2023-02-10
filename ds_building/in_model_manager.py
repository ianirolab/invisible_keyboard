import tensorflow as tf
import os
import shutil
import pickle

def getBaseModel():
    return tf.keras.models.load_model('model_s_500-500-500-200-20')

def getModel0():
    return tf.keras.models.load_model('model-0-70')

def getModel1():
    return tf.keras.models.load_model('model-0-74')

def getTempModel():
    return tf.keras.models.load_model('./model-temp/model-temp-1300')


def dataset_merger():
    id = 11624
    if 'x' not in os.listdir():
        os.mkdir('x')
        os.mkdir('y')

    # for d in os.listdir('./history/x'):
    for d in ['x-0-60', 'x-0-61']:
        for f in os.listdir('./history/x/' + d):
            shutil.copyfile('./history/x/' + d + '/' + f,'./x/' + str(id))
            shutil.copyfile('./history/y/y' + d[1:] + '/'+ f,'./y/' + str(id))
            id += 1

def convert():
    for d in ('x-0-50',):
        for f in os.listdir('./history/x/' + d):
            with open('./history/x/' + d + '/' + f,'rb') as fl:
                x = pickle.load(fl)
            with open('./history/y/y' + d[1:]+ '/' + f,'rb') as fl:
                y = pickle.load(fl)
dataset_merger()
