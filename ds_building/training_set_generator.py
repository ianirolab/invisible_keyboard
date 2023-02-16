# Creates a dataset from results folder and pushing.txt file

import os
import shutil
import pickle

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


if 'x' not in os.listdir(): 
    os.mkdir('./ds_building/inputs/x')
    os.mkdir('./ds_building/inputs/y')

for i in range(len(lines)):
    shutil.copy('./ds_building/inputs/results/' + lines[i][0],'./ds_building/inputs/x')
    arr = [0 for i in range(10)]
    arr[int(lines[i][1])] = 1

    with open('./ds_building/inputs/y/' + lines[i][0],'wb') as f:
        pickle.dump(arr,f)

        

