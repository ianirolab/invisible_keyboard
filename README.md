# Camera keyboard
## _Tester setup:_

According to the input device, select one of the cap initializations from:
```python
cap = cv2.VideoCapture(0)                           #recommended
cap = cv2.VideoCapture('./path_to_video_file')
```

Camera must be in selfie view (image flipped) for the tester to work and in order for the paths to work, the root folder must always be project's root folder.

# Testing:
##### ./key_tester.py: 
Runs camera input through both the finger recognition and the key recognition models.
##### ./ds_building/finger_tester.py: 
Runs camera input only through the finger recognition model.

Testers won't save any data, they only provide a plug and play way of testing live (or through a saved video) the neural networks.
Testers will predict in the top left corner the key that is being pushed, as well as detecting the status of "raised hands".The neural network(s) will be triggered when both hands are visible by mediapipe.

### Neural network pipeline
To get the predicted key, the  pipeline looks like this:
+ Take frame from video or webcame
+ Process the frame with MediaPipe to get 21 points (x,y,z) per hand
+ Process the 42 points with FingerModel to get a prediction on the finger that is pushing
+ Process the 21 points of the pushing hand with KeyModel 
+ Get a prediction on the key that is being pushed

### Project Structure and Data Gathering Routine
The project can be divided in two main sections: **Finger** and the **Key** sections.
Although the neural networks in the two sections have a slightly different structure and work differently, both sections share pretty much the same structure for the dataset building.
The data gathering starts with the **_helper** file, which records frames from the camera, while the user can see in live what the latest neural network trained predicts. This helps with finding hand positions unknown to the neural network.
Data is sent to the ./inputs/**pictures** and ./inputs/**results** directories. From here the **_picker** file has a simple classification interface (for humans). Now the data is stored in the ./inputs/**x1** and ./inputs/**y1** directories, where it can be merged to directories ./inputs/**x** and ./inputs/**y** using (manually) function **_merger()** in **in_model_manager.py**. 
In the **x** folder, there will be pickled files containing hand(s) points, while the **y** folder contains pickled files with:
+ **Finger**: 1-hot encoded arrays of size 10 (1 place for the status "raised", 1 for the thumbs, 1 place for each of the remaining fingers)
+ **Key**: the key that has been pushed (it will be converted to a 1-hot encoded array during the dataset loading)


### Training
Once the data has been gathered and stored in directories **x** and **y**, as well as the testing data in directories **x_test** and **y_test**, running the **_nn** file will start the training. During the training accuracy and loss graphs will be plotted every 50 epochs, but that can be easily changed.
```python
if(epoch % 50 == 0):
            plot(self.l,self.name)
```
The **key_nn.py** file can train all 8 neural networks in one run, sequentially. Refer to the graph plotted while training to know which neural networks is being trained; however all plots will be saved int the **plots** folder 
