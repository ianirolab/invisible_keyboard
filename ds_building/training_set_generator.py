# Creates a dataset from results folder and pushing.txt file
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import shutil
import pickle
from in_model_manager import db_setup

# xdir = 'x1'
# ydir = 'y1'
xdir = 'x-test1'
ydir = 'y-test1'

db_setup(xdir,ydir)

lines = []
with open('./ds_building/pushing.txt','r') as f:
    lines = f.readlines()

for i in range(len(lines)):
    lines[i] = lines[i].strip().split()
    idx = lines[i][0].index('.')
    lines[i][0] = lines[i][0][:idx]
    
ct = 0
for n in os.listdir('./ds_building/inputs/results'):
    flag = False
    for l in lines:
        if l[0] == n:
            flag = True
            break

    if not flag:
        lines.append([n,9])
        ct += 1

for i in range(len(lines)):
    shutil.copy('./ds_building/inputs/results/' + lines[i][0],'./ds_building/inputs/'+xdir+'/' + lines[i][0])
    
    arr = [0 for i in range(10)]
    arr[int(lines[i][1])] = 1

    with open('./ds_building/inputs/'+ydir+'/' + lines[i][0],'wb') as f:
        pickle.dump(arr,f)

        

