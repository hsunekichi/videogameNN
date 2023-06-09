from tensorflow import keras
import numpy as np
import data_capture.data_capture as data_capture
import pyautogui as autogui
#import keyboard
import config as conf
from model_functions import model_execution as model_exec
from data_processing import image_preproc as img_preproc
import model_functions.model_definitions as models
import sys
import pickle as pk
from sklearn.decomposition import PCA
import time

autogui.PAUSE = 0
autogui.MINIMUM_SLEEP = 0.0


# Define the keys and their corresponding indices in the list
key_mapping = {
    #'left': 0,
    #'right': 1,
    #'down': 2,
    #'up': 3,
    #'x': 4,
    'z': 0,
    #'a': 6,
    #'c': 7
}

def send_input(key_states):
    # Send keyboard input based on the values in the list
    for key, index in key_mapping.items():
        if key_states[index]:
            autogui.keyDown(key)
        else:
            autogui.keyUp(key)

if (len(sys.argv) < 3):
    print("Usage: python run_model.py <dataset_name> <model_id> [<execute_inputs>]")
    exit()

name = sys.argv[1]
id = sys.argv[2]

if (len(sys.argv) < 4):
    execute_inputs = True
else:
    execute_inputs = (sys.argv[3] == "1")

model = keras.models.load_model("trained_models/modelos_"+name+"/modelo_"+id+"_completo.h5",
                                custom_objects={'loss': model_exec.ChangeBinaryCrossentropy(),
                                                'ImageBuffer': models.ImageBuffer})
#pca = pk.load(open("trained_models/modelos_"+name+"/pca.pkl",'rb'))
pca = None


# Get the first frame to initialize the window
img = data_capture.getScreen()
    
# Preprocess the image
img = img_preproc.downscale_images(img, conf.scale)
frame = img_preproc.reduce_frames(img, pca)    


#frame_array = np.empty((conf.n_timesteps, 
#                        np.shape(frame)[0], np.shape(frame)[1], 1), conf.pixel_type)

while True:
    
    init = time.time_ns()
    img = data_capture.getScreen()
    
    # Preprocess the image
    img = img_preproc.downscale_images(img, conf.scale)
    frame = img_preproc.reduce_frames(img, pca)    

    """
    # Add the frame to the window
    frame_array = np.append(frame_array, [frame], axis=0)

    # Remove the first frame if the window is full
    if len(frame_array) > conf.n_timesteps:
        frame_array = np.delete(frame_array, 0, axis=0)
    """
    
    # Predict the output
    prediction = model_exec.predict(model, frame)
    
    # Execute the prediction
    print(prediction)

    if (execute_inputs):
        send_input(prediction)

    # Exit the loop when the 'q' key is pressed
    #if autogui.is_pressed('q'):
    #    break

    end = time.time_ns()
    
    print("Ms: ", (end-init)/1000000)

    # Calculate time to sleep to maintain the fps cap
    elapsed_time = end - init
    sleep_time = max(1/conf.record_fps - elapsed_time, 0)
    time.sleep(sleep_time)




