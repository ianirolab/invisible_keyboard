import matplotlib.pyplot as plt
import pickle

with open('train_h','rb') as f:
    train_h = pickle.load(f)


plt.plot(train_h.history['accuracy'])
plt.title('Train accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

plt.plot(train_h.history['loss'])
plt.title('Train loss')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()
