import matplotlib.pyplot as plt
import pickle
from multiprocessing import Process,Queue
from scipy.signal import savgol_filter

# with open('train_h','rb') as f:
#     train_h = pickle.load(f)

data = {}

qa = Queue(100)
process = None

def plot(pdata):
    global process
    if(process is not None):
        process.kill()
    qa.put(pdata)
    process = Process(target=plot_graph_acc)
    process.start()
    

def plot_graph_acc(*args):
    data = qa.get()


    plt.title('Train loss ' + str(len(data['loss'])-1))
    plt.plot(data['loss'],label ='train loss')
    plt.plot(data['val_loss'],label ='validation loss')
    # plt.plot(data['mytest-los'],label ='test loss')
    ya = savgol_filter(data['mytest-los'], 20, 3) # window size 51, polynomial order 3
    plt.plot(ya,label ='rounded test loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.figure()

    plt.title('Train accuracy ' + str(len(data['loss'])-1))
    plt.plot(data['categorical_accuracy'],label ='train accuracy')
    plt.plot(data['val_categorical_accuracy'],label ='validation accuracy')
    ya = savgol_filter(data['mytest-acc'], 20, 3) # window size 51, polynomial order 3
    plt.plot(ya,label ='rounded test accuracy')
    # plt.plot(data['mytest-acc'],label ='test accuracy')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.show()
    exit() 

   
