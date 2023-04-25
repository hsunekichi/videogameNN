from inputs import get_gamepad
import math
import threading
import time
import sys
import cv2
import numpy as np
import data_capture.data_capture as data_capture 
import config as conf
import glob 
import os



nombre = sys.argv[1]

if (len(sys.argv) < 3):
    escribir_logs = True
else:
    escribir_logs = bool(sys.argv[2] == "1")

if not os.path.exists("datasets/"+nombre+"/logs"):
    os.makedirs("datasets/"+nombre+"/logs")

if not os.path.exists("datasets/"+nombre+"/videos"):
    os.makedirs("datasets/"+nombre+"/videos")


# Especifica el patrón de nombres de archivo que quieres buscar
pattern = "datasets/"+nombre+'/videos/vid*.mp4'

# Usa glob para obtener una lista de nombres de archivo que cumplan el patrón
files = glob.glob(pattern)

# Obtener el ID más alto
max_id = -1
max_id_file = ""
for file in files:
    if (file.split("part")[0] == file):
        number = file.split("videos/vid")[1].split(".mp4")[0]
    else:
        number = file.split("videos/vid")[1].split("_part")[0]

    file_id = int(number)
    if file_id > max_id:
        max_id = file_id
        max_id_file = file

indice_video = str(max_id+1)

logFile = "datasets/"+nombre+"/logs/log"+indice_video+".txt"
vidFile = "datasets/"+nombre+"/videos/vid"+indice_video+".mp4"



polling = 16                # Tiempo de refresco en ms


#screen_size = tuple(pyautogui.size()) # Set the screen size to match your monitor's resolution
screen_size = (conf.screen_width, conf.screen_height)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
outImg = cv2.VideoWriter(vidFile, fourcc, conf.record_fps, screen_size)
joy = data_capture.XboxController()

polling = polling/1000      # Calcula el tiempo de refresco en segundos


f = open(logFile, "w+")

print("Recording\n")
if (escribir_logs):
    sys.stdout = f


try:
    while True:
        start_time = time.time()

        print(data_capture.getInput(joy))
        
        if escribir_logs:
            img = data_capture.getScreen()
            outImg.write(img)
        
        end_time = time.time()

        # Calculate time to sleep to maintain the fps cap
        elapsed_time = end_time - start_time
        sleep_time = max(1/conf.record_fps - elapsed_time, 0)
        time.sleep(sleep_time)


        #print("Tiempo de refresco: ", end_time - start_time, "s")

finally:
    outImg.release() 
    f.close()   

