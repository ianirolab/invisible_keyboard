import matplotlib.pyplot as plt
import pickle
from multiprocessing import Process,Queue
from scipy.signal import savgol_filter

# with open('train_h','rb') as f:
#     train_h = pickle.load(f)

data = {}

qa = Queue(100)
process = None

def plot(pdata,name):
    global process
    if(process is not None):
        process.kill()
    qa.put(pdata)
    process = Process(target=plot_graph_acc,args=name)
    process.start()
    

def plot_graph_acc(*args):
    name = ''.join(args)
    data = qa.get()


    plt.title(name + ' train loss ' + str(len(data['loss'])-1))
    plt.plot(data['loss'],label ='train loss')
    # plt.plot(data['val_loss'],label ='validation loss')
    # plt.plot(data['mytest-los'],label ='test loss')
    ya = savgol_filter(data['val_loss'], 20, 3)
    plt.plot(ya,label ='rounded validation loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.figure()
    plt.savefig('./plots/'+name+'/loss.png')

    plt.title(name + ' train accuracy ' + str(len(data['loss'])-1))
    plt.plot(data['categorical_accuracy'],label ='train accuracy')
    # plt.plot(data['val_categorical_accuracy'],label ='validation accuracy')
    ya = savgol_filter(data['val_categorical_accuracy'], 20, 3) 
    plt.plot(ya,label ='rounded test accuracy')
    # plt.plot(data['mytest-acc'],label ='test accuracy')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig('./plots/'+name+'/accuracy.png')
    

    plt.show()
    exit() 

   
def plot_hands():
    with open('results/sx_ind/22','rb') as f:
        a = pickle.load(f)

    xs = []
    ys = []
    zs = []
    xs2 = []
    ys2 = []
    zs2 = []
    for i in range(0,len(a),6):
        xs.append(a[i])
    for i in range(1,len(a),6):
        ys.append(a[i])
    for i in range(2,len(a),6):
        zs.append(a[i])
    for i in range(3,len(a),6):
        xs2.append(a[i])
    for i in range(4,len(a),6):
        ys2.append(a[i])
    for i in range(5,len(a),6):
        zs2.append(a[i])

    ax = plt.figure().add_subplot(projection='3d')
    ax.scatter(xs,ys,zs)
    ax2 = plt.figure().add_subplot(projection='3d')
    ax2.scatter(xs2,ys2,zs2)
    plt.show()

# plot_hands()