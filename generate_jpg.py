import cv2
import os
import sys
#from PIL import Image
from numba import jit, cuda
import numpy as np
import math
import time
import data_processing.data_generation as data_generation
import data_processing.image_postproc as image_postproc

## ------------------ MAIN ------------------ ##

dirName = sys.argv[1]
which_videos = sys.argv[2]

scale = int(sys.argv[3])
path = 'frames_'+dirName

# Create the output folder if it doesn't exist
if not os.path.exists(path):
    os.makedirs(path)

print("datasets/"+dirName+"/videos/vid"+which_videos+".mp4")
bw_images = data_generation.load_all_frames("datasets/"+dirName+"/videos/vid"+which_videos+".mp4", scale)

denormalized_images = image_postproc.regenerate_images(bw_images)
total_frames = len(bw_images)
count = 0
for image in denormalized_images:
    cv2.imwrite(os.path.join(path, f'{count:04d}.jpg'), image)  # Save the frame as a JPEG image

    count += 1
    
    # Calculate the progress as a percentage
    progress = count / total_frames * 100
    
    # Print the progress bar
    sys.stdout.write('\r')
    sys.stdout.write(f'Saving frames: [{int(progress):3d}%] {"#"*int(progress/2)}{"-"*int(50-progress/2)}')
    sys.stdout.flush()