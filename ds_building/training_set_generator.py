import os
import shutil
import pickle

lines = []
with open('pushing.txt','r') as f:
    lines = f.readlines()

for i in range(len(lines)):
    lines[i] = lines[i].strip().split()
    idx = lines[i][0].index('.')
    lines[i][0] = lines[i][0][:idx]
    
ct = 0
for n in os.listdir('results'):
    # if ct == 70:
    #     break
    flag = False
    for l in lines:
        if l[0] == n:
            flag = True
            break

    if not flag:
        lines.append([n,9])
        ct += 1


if 'x' not in os.listdir(): 
    os.mkdir('./x')
    os.mkdir('./y')

for i in range(len(lines)):
    shutil.copy('./results/' + lines[i][0],'./x')
    arr = [0 for i in range(10)]
    arr[int(lines[i][1])] = 1

    with open('./y/' + lines[i][0],'wb') as f:
        pickle.dump(arr,f)

        

