# Invisible keyboard
# Camera setup: 
Connect android phone (with usb debugging active) with IP camera via USB and run
adb forward tcp:5555 tcp:IP_CAMERA_PORT

#Workflow for pushing_nn:
- After connecting the camera, run cam_through_mp2.py, with that 1000 frames with both hands appearing will be saved to ./pictures.
- Use image_picker.py to remove a picture(2), mark it as pushing(1), mark it as raised (any key - 0). With left arrow key, it's possible to go back in case of error. This will generate pushing.txt
- Use training_set_generator.py to create folders x and y that will respectively hold inputs and outputs for the nn. Output files are based on pushing.txt

- Run the training with pushing_nn
- Finally have a look at the results plotted with graphs.py and test the nn with pushing_tester.py
