import config as conf
import numpy as np
from tensorflow.keras.utils import Sequence
import tensorflow as tf
import cv2
import data_processing.image_preproc as image_preproc
import data_processing.input_preproc as input_preproc
import sys
import glob




########################################################
## ------------------ Data loading ------------------ ##
########################################################

# Loads all the frames of several videos 
def load_all_frames(video_reg_exp, scale = conf.scale, pca = None):
    
        # Gets all the files that match the regular expression
        files = glob.glob(video_reg_exp)
    
        # Sorts the files by name
        files.sort()

        # Loads all the frames from all the videos
        print("\nLoading frames from ", files[0], "\n")
        frames = load_one_video_frames(files[0], scale, pca = pca)
        
        for file in files[1:]:
            print("\nLoading frames from ", file, "\n")
            new_frames = load_one_video_frames(file, scale, pca = pca)
            frames = np.append(frames, new_frames, axis=0)
    
        return frames



# Loads all the inputs of several files
def load_all_inputs(input_reg_exp):
    
        # Gets all the files that match the regular expression
        files = glob.glob(input_reg_exp)
    
        # Sorts the files by name
        files.sort()
    
        # Loads all the frames from all the videos
        print("\nLoading inputs from ", files[0], "\n")
        inputs = load_one_file_inputs(files[0])

        for file in files[1:]:
            print("\nLoading inputs from ", file, "\n")
            inputs = np.append(inputs, load_one_file_inputs(file), axis=0)

        # Removes the first dimension of the array
        return inputs



# Returns an array with the frames of a video, unprocessed
def load_real_frames(vidName):
    
    # Open the video file
    vidcap = cv2.VideoCapture(vidName)

    # Create the output folder if it doesn't exist
    #if not os.path.exists(path):
    #    os.makedirs(path)

    # Get the total number of frames in the video
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    image_array = np.empty((total_frames, int(conf.screen_height), int(conf.screen_width), 3), dtype=np.uint8)
    count = 0
    success, image = vidcap.read()              # Gets a new frame
    # Reads all frames
    while success:

        image_array[count] = image              # Adds the frame to the array
        success, image = vidcap.read()          # Gets a new frame

        count += 1

        # Calculate the progress as a percentage
        progress = count / total_frames * 100

        # Print the progress bar
        sys.stdout.write('\r')
        sys.stdout.write(f'Loading: [{int(progress):3d}%] {"#"*int(progress/2)}{"-"*int(50-progress/2)}')
        sys.stdout.flush()

    return image_array



# Loads all the frames of a given video and returns them processed (downscaled, greyscale...)
def load_one_video_frames(vidName, scale = conf.scale, pca = None):

    # Open the video file
    vidcap = cv2.VideoCapture(vidName)

    # Create the output folder if it doesn't exist
    #if not os.path.exists(path):
    #    os.makedirs(path)

    # Get the total number of frames in the video
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    image_array = np.empty((total_frames, int(conf.screen_height/scale), int(conf.screen_width/scale), 3), dtype=np.uint8)
    count = 0
    success, image = vidcap.read()              # Gets a new frame
    # Reads all frames
    while success:
        image = image_preproc.downscale_images(image, scale)
        image_array[count] = image
        success, image = vidcap.read()          # Gets a new frame

        count += 1

        # Calculate the progress as a percentage
        progress = count / total_frames * 100

        # Print the progress bar
        sys.stdout.write('\r')
        sys.stdout.write(f'Loading: [{int(progress):3d}%] {"#"*int(progress/2)}{"-"*int(50-progress/2)}')
        sys.stdout.flush()

    bw_array = image_preproc.reduce_frames(image_array, pca)

    return bw_array[:len(bw_array)-1] # Removes the last frame to align it with the inputs   



# Loads the inputs from a file and discretizes them
def load_one_file_inputs(path):
    data = []

    with open(path) as f:
        for line in f:
            # Strip off any whitespace from the beginning and end of the line
            line = line.strip()

            # Parse the line as a list of numbers
            line_data = [float(x) for x in line[1:-1].split(",")]

            processed_line = input_preproc.discretize_line_input(line_data)

            # Add the parsed data to the list
            data.append(processed_line)
    
    # Convert the list to a numpy.ndarray
    data = np.array(data)
    
    return data[1:]




###########################################################
## ------------------ Data generation ------------------ ##
###########################################################

# Generates a sliding window of images and outputs and size `window_size`
def sliding_window(images, outputs, window_size=conf.n_timesteps):
    """
    Given an array of images and an array of outputs, returns a sliding window of size `window_size` 
    that pairs 10 consecutive images with the output of the last one.
    
    Args:
    images (ndarray): An array of images. The first dimension represents the number of images, 
                      and the remaining dimensions are the shape of each image.
    outputs (ndarray): An array of outputs. This should have the same number of rows as `images`.
    window_size (int): The size of the sliding window. Defaults to 10.
    
    Returns:
    A tuple of two arrays:
    - `X`: An array of shape `(num_windows, window_size, *image_shape)`, representing the sliding window 
           of images.
    - `y`: An array of shape `(num_windows, *output_shape)`, representing the corresponding outputs for 
           each sliding window.
    """
    num_images, *image_shape = images.shape
    num_windows = num_images - window_size + 1
    
    # Create empty arrays to hold the sliding window data
    X = np.zeros((num_windows, window_size, *image_shape), dtype=images.dtype)
    y = np.zeros((num_windows, *outputs[0].shape), dtype=outputs.dtype)

    # Iterate over the sliding window and populate `X` and `y`
    for i in range(num_windows):
        X[i] = images[i:i+window_size]
        y[i] = outputs[i+window_size-1]
        
    return X, y



# A Keras Sequence that generates batches of sliding windows of images and outputs and size `window_size`
class SlidingWindowGenerator(Sequence):
    def __init__(self, images, outputs, window_size=conf.n_timesteps, batch_size=32):
        self.images = images
        self.outputs = outputs
        self.window_size = window_size
        self.batch_size = batch_size
        self.num_images, *self.image_shape = images.shape
        self.num_windows = self.num_images - self.window_size + 1
    
    def __len__(self):
        return int(tf.math.ceil(self.num_windows / self.batch_size))
    
    
    def __getitem__(self, idx):
        batch_start = idx * self.batch_size
        batch_end = (idx + 1) * self.batch_size
        
        if batch_end > self.num_windows:
            batch_end = self.num_windows

        batch_images = np.zeros((batch_end - batch_start, self.window_size, *self.image_shape), dtype=self.images.dtype)
        batch_outputs = np.zeros((batch_end - batch_start, *self.outputs[0].shape), dtype=self.outputs.dtype)
        
        for i in range(batch_start, batch_end):
            batch_images[i - batch_start] = self.images[i:i+self.window_size]
            batch_outputs[i - batch_start] = self.outputs[i+self.window_size-1]
        
        return batch_images, batch_outputs    
    
"""
    def __getitem__(self, idx):
        batch_start = idx * self.batch_size
        batch_end = (idx + 1) * self.batch_size
        
        if batch_end > self.num_windows:
            batch_end = self.num_windows
        
        print()
        print()
        print(np.shape(self.images))
        print(np.shape(self.images[batch_start:batch_start+self.window_size*(batch_end - batch_start)]))
        num_windows_in_batch = batch_end - batch_start
        batch_images = np.reshape(self.images[batch_start:batch_start+self.window_size*num_windows_in_batch],
            (num_windows_in_batch, self.window_size, *self.image_shape))
        batch_outputs = self.outputs[batch_start+self.window_size-1:batch_end]
    
        return batch_images, batch_outputs
"""


#################################################################################
## ---------------------------- Private functions ---------------------------- ##
#################################################################################

