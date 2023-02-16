import tensorflow as tf
import os
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
def db_setup(x_dir = 'x1', y_dir = 'y1'):
    if(x_dir not in os.listdir()):
        os.mkdir(x_dir)
    if(y_dir not in os.listdir()):
        os.mkdir(y_dir)
        
    if('./ds_building/inputs/'+x_dir+'' not in os.listdir()):
        os.mkdir('./ds_building/inputs/'+x_dir+'')
        os.mkdir('./ds_building/inputs/'+y_dir+'')

def key_db_setup(x_dir = 'x1', y_dir = 'y1'):
        
    # x_dir = 'pictures'
    # y_dir = 'results'
    
    if(x_dir not in os.listdir()):
        os.mkdir(x_dir)
    if(y_dir not in os.listdir()):
        os.mkdir(y_dir)
    for s in subf:
        os.mkdir('inputs/'+x_dir+'/'+s)
        os.mkdir('inputs/'+y_dir+'/'+s)
#
#
########################################################################
#
# Utilities
# 

def key_merger():
    xdir = 'x-test1'
    ydir = 'y-test1'
    dest_dirx = 'x-test'
    dest_diry = 'y-test'
    for dir in os.listdir(xdir):
        if len(os.listdir(dest_dirx+'/'+dir)) > 0:
            ids = [int(i) for i in os.listdir(dest_dirx+'/'+dir)]
            id = max(ids) + 1
        else:
            id = 0
        
        for f in os.listdir(xdir+'/'+dir):
            shutil.copy('inputs/'+xdir+'/'+dir+'/'+f, 'inputs/'+dest_dirx+'/'+dir+'/'+str(id))
            shutil.copy('inputs/'+ydir+'/'+dir+'/'+f, 'inputs/'+dest_diry+'/'+dir+'/'+str(id))
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
# key_db_setup_test()
