import tensorflow as tf
import os
import shutil
import pickle

def getBaseModel():
    return tf.keras.models.load_model('model_s_500-500-500-200-20')

def getModel0():
    return tf.keras.models.load_model('model-0-70')

def getModel1():
    return tf.keras.models.load_model('Model')

def getMidModel():
    return tf.keras.models.load_model('./ds_building/Model')

def getTempModel():
    return tf.keras.models.load_model('./model-temp/model-temp-800')

def getKeyModel(name):
    return tf.keras.models.load_model('./key_models/model-0-83/'+name)


def dataset_merger():
    for f in os.listdir('./x2'):
        shutil.copyfile('./x2/' + f,'./x/' + f)
        shutil.copyfile('./y2/' + f,'./y/' + f)

def convert():
    for d in ('x-0-50',):
        for f in os.listdir('./history/x/' + d):
            with open('./history/x/' + d + '/' + f,'rb') as fl:
                x = pickle.load(fl)
            with open('./history/y/y' + d[1:]+ '/' + f,'rb') as fl:
                y = pickle.load(fl)

def key_db_setup():
    if('x2' not in os.listdir()):
        os.mkdir('x2')
        os.mkdir('y2')
    subf = ('sx_mig','sx_anu','sx_mid','sx_ind','thumb','dx_ind','dx_mid','dx_anu','dx_mig','raised')
    for s in subf:
        os.mkdir('x2/'+s)
        os.mkdir('y2/'+s)

def key_db_setup_test():
    if('x-test3' not in os.listdir()):
        os.mkdir('x-test3')
        os.mkdir('y-test3')
    subf = ('sx_mig','sx_anu','sx_mid','sx_ind','thumb','dx_ind','dx_mid','dx_anu','dx_mig','raised')
    for s in subf:
        os.mkdir('x-test3/'+s)
        os.mkdir('y-test3/'+s)

def key_pics_setup():
    if('pictures' not in os.listdir()):
        os.mkdir('pictures')
        os.mkdir('results')
    subf = ('sx_mig','sx_anu','sx_mid','sx_ind','thumb','dx_ind','dx_mid','dx_anu','dx_mig','raised')
    for s in subf:
        os.mkdir('pictures/'+s)
        os.mkdir('results/'+s)

def key_merger():
    xdir = 'x-test3'
    ydir = 'y-test3'
    dest_dirx = 'x-test2'
    dest_diry = 'y-test2'
    for dir in os.listdir(xdir):
        if len(os.listdir(dest_dirx+'/'+dir)) > 0:
            ids = [int(i) for i in os.listdir(dest_dirx+'/'+dir)]
            id = max(ids) + 1
        else:
            id = 0
        
        for f in os.listdir(xdir+'/'+dir):
            shutil.copy(xdir+'/'+dir+'/'+f,dest_dirx+'/'+dir+'/'+str(id))
            shutil.copy(ydir+'/'+dir+'/'+f,dest_diry+'/'+dir+'/'+str(id))
            id+=1



# key_db_setup()
# key_pics_setup()
key_merger()
# key_db_setup_test()
