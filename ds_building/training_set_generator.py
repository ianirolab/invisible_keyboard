import os
import shutil
import pickle


lines = []
with open('pushing.txt','r') as f:
    lines = f.readlines()

for i in range(len(lines)):
    lines[i] = lines[i].strip()


if 'x' not in os.listdir(): 
    os.mkdir('./x')
    os.mkdir('./y')

for p in os.listdir('pictures'):
    idx = p.index('.')
    fname = p[:idx]
    shutil.copy('./results/' + fname,'./x')
    with open('./y/' + fname,'wb') as f:
        pickle.dump(p in lines,f)
        

