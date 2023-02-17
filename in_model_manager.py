import os
import tensorflow as tf
import shutil

# subfolders for any input datasets 
subf = ('sx_mig','sx_anu','sx_mid','sx_ind','thumb','dx_ind','dx_mid','dx_anu','dx_mig','raised')

########################################################################
# 
# Model retrieving functions
#
def getTempModel_ds(n=800):

    return tf.keras.models.load_model('./ds_building/model-temp/model-temp-'+n)

def getFingerModel():
    return tf.keras.models.load_model('./ds_building/Model')

def getTempModel(n=800):
    return tf.keras.models.load_model('./model-temp/model-temp-'+n)

def getKeyModel(name):
    return tf.keras.models.load_model('./KeyModel/'+name)
#
#
########################################################################
# 
# Database setup
#

def db_starter():
    if ('inputs' not in os.listdir()):
        os.mkdir('./inputs')
    if ('inputs' not in os.listdir('ds_building')):
        os.mkdir('./ds_building/inputs')
    if ('plots' not in os.listdir()):
        os.mkdir('./plots')
    for f in subf:
        if (subf not in os.listdir('./plots')):
            os.mkdir('./plots/'+subf)
    
    if ('finger' not in os.listdir('./plots')):
        os.mkdir('./plots/finger')
        
    
def db_setup(x_dir = 'x1', y_dir = 'y1'):
    if(x_dir not in os.listdir('./ds_building/inputs')):
        os.mkdir('./ds_building/inputs/'+x_dir)
        os.mkdir('./ds_building/inputs/'+y_dir)
        

def key_db_setup(x_dir = 'x1', y_dir = 'y1'):
        
    # x_dir = 'pictures'
    # y_dir = 'results'
    
    if(x_dir not in os.listdir('./inputs')):
        os.mkdir('./inputs/' + x_dir)
        os.mkdir('./inputs/' + y_dir)
    for s in subf:
        if s not in os.listdir('./inputs/'+x_dir):
            os.mkdir('./inputs/'+x_dir+'/'+s)
            os.mkdir('./inputs/'+y_dir+'/'+s)
#
#
########################################################################
#
# Utilities
# 

def finger_merger():
    FINGER = True
    inputsPath = './ds_building/inputs/' if FINGER else './inputs/'
    # xdir = 'x1'
    # ydir = 'y1'
    # dest_dirx = 'x'
    # dest_diry = 'y'
    xdir = 'x-test1'
    ydir = 'y-test1'
    dest_dirx = 'x-test'
    dest_diry = 'y-test'

    if dest_dirx not in os.listdir(inputsPath):
        os.mkdir(inputsPath + dest_dirx )
        os.mkdir(inputsPath + dest_diry )

    id = 0
    if len(os.listdir(inputsPath + dest_dirx)) > 0:
        ids = [int(i) for i in os.listdir(inputsPath + dest_dirx)]
        id = max(ids) + 1
    
    
    for f in os.listdir(inputsPath + xdir):
        shutil.copy(inputsPath + xdir+'/'+f, inputsPath + dest_dirx+'/'+str(id))
        shutil.copy(inputsPath + ydir+'/'+f, inputsPath + dest_diry+'/'+str(id))
        id+=1

def key_merger():
    inputsPath = './inputs/'
    # xdir = 'x1'
    # ydir = 'y1'
    # dest_dirx = 'x'
    # dest_diry = 'y'
    xdir =  'x-test1'
    ydir =  'y-test1'
    dest_dirx =  'x-test'
    dest_diry =  'y-test'

    key_db_setup(dest_dirx, dest_diry)

    for dir in os.listdir(inputsPath + xdir):
        if len(os.listdir(inputsPath + dest_dirx+'/'+dir)) > 0:
            ids = [int(i) for i in os.listdir(inputsPath + dest_dirx+'/'+dir)]
            id = max(ids) + 1
        else:
            id = 0
        
        for f in os.listdir(inputsPath + xdir+'/'+dir):
            shutil.copy(inputsPath + xdir+'/'+dir+'/'+f, inputsPath + dest_dirx+'/'+dir+'/'+str(id))
            shutil.copy(inputsPath + ydir+'/'+dir+'/'+f, inputsPath + dest_diry+'/'+dir+'/'+str(id))
            id+=1
#
#
####################################
#
# Manual function calls 
#
# key_db_setup()
# key_pics_setup()
# key_merger()
# finger_merger()
# key_db_setup_test()
# db_starter()
