# Camera keyboard
# Tester setup:

According to the type of input, select one of the 3 cap initializations in key_tester. Camera must be in selfie view (image flipped) for the tester to work

# Tester files:

key_tester.py: runs camera input through both the finger recognition and the key recognition models 
ds_building/finger_tester.py: runs camera input only through the finger recognition model 

Testers won't save any data, they only provide a way of testing live (or through a saved video) the neural networks. 
Testers will predict in the top left corner the key that is being pushed, as well as detecting the status of "raised hands".The neural network(s) will be triggered when both hands are visible by mediapipe