import numpy as np
from numba import cuda
import sys
import sys
import math
import config as conf
import cv2
import time


# Reduces the dimensionality of the images using several techniques, in order:
# Converts to grayscale, and erases other channels (returns only one)
# Normalizes pixels between 0 and 1
# Applies PCA
def reduce_frames (img_array, PCA = None):

    one_frame = False

    if len(np.shape(img_array)) == 3:   # Only one frame
        img_array = np.expand_dims(img_array, axis=0)
        one_frame = True


    nImages = len(img_array)

    heigth = img_array[0].shape[0]
    width = img_array[0].shape[1]

    bw_images = np.empty((nImages, heigth, width, 1), dtype=conf.pixel_type)

    # Create CUDA streams for each image
    streams = [cuda.stream() for _ in range(nImages)]

    # Loops through the images and launch kernels in parallel
    for i in range(nImages):

        # Allocate device memory for the processed image
        gpu_bw_array = cuda.device_array((heigth, width, 1), dtype=conf.pixel_type)

        # Calculate the progress as a percentage
        progress = (i+1) / nImages * 100

        # Print the progress bar
        #sys.stdout.write('\r')
        #sys.stdout.write(f'Processing: [{int(progress):3d}%] {"#"*int(progress/2)}{"-"*int(50-progress/2)}')
        #sys.stdout.flush()

        # Moves the image to GPU in a new stream
        gpu_image = cuda.to_device(img_array[i], stream=streams[i])

        # calculate the block and grid dimensions
        grid_dim = (gpu_image.shape[0] // conf.block_dim[0] + 1, gpu_image.shape[1] // conf.block_dim[1] + 1)

        # invoke the kernel to convert the image to black and white in the same stream
        __convert_to_bw_kernel[grid_dim, conf.block_dim](gpu_image, gpu_bw_array)

        # Copy the processed image back to host memory
        cuda.synchronize()
        gpu_bw_array.copy_to_host(bw_images[i])


    #bw_images = (img_array.astype(conf.pixel_type))/255
    
    if PCA != None: # If there is a PCA object, uses it to reduce the images
        bw_images = reduce_images_pca(bw_images, PCA, conf.downscale_batch_size)

    if one_frame:
        return bw_images[0]
    else:
        return bw_images



# Reduces the dimensionality of the images using PCA
def reduce_images_pca(images, model, batch_size = 0):
    """
    if batch_size == 0:
        img_array = np.reshape(images, (1, np.shape(images)[0]*np.shape(images)[1]))
        reduced = PCA.transform(img_array)
        reduced_images = np.reshape(reduced, (conf.PCA_image_side, conf.PCA_image_side, 1))

        return reduced_images
    
    else:
        print()
        print(f'Downsampling with PCA {len(images)} images...')

        n_batches = math.ceil(len(images)/batch_size)

        img_array = np.reshape(images, (np.shape(images)[0], np.shape(images)[1]*np.shape(images)[2]))
        reduced_array = np.empty((len(img_array), conf.PCA_image_side*conf.PCA_image_side), dtype=conf.pixel_type)
        
        for i in range(n_batches):
            init = i*batch_size
            end = ((i+1)*batch_size)-1
            print(f'Batch {i+1}/{n_batches} ({init}-{end})')
            img_batch = img_array[init:end]
            reduced_array[init:end] = PCA.transform(img_batch)
        
        reduced_images = np.reshape(reduced_array, (np.shape(reduced_array)[0], conf.PCA_image_side, conf.PCA_image_side, 1))

        return reduced_images
    """
    
    print("Reducing images with autoencoder...")
    reformed = model.predict(images)

    return reformed


# Downscale the images to a smaller size using a cv2 interpolation algorithm
def downscale_images(img_array, scale = conf.scale):
    
    # If there is only one image, there is no array to loop through
    if len(np.shape(img_array)) == 3:
        return cv2.resize(img_array, (int(conf.screen_width/scale), int(conf.screen_height/scale)))
    else:
        reduced_images = np.empty((
                                    len(img_array), 
                                    int(conf.screen_height/scale), 
                                    int(conf.screen_width/scale), 
                                    conf.n_channels
                                ),
                                dtype=conf.pixel_type)

        for i in range(len(img_array)):
            reduced_images[i] = cv2.resize(img_array[i], (int(conf.screen_width/scale), int(conf.screen_height/scale)))

        return reduced_images



# Trims the images to remove the borders
def trim_images(img_array):

    if len(np.shape(img_array)) == 3:
        return img_array[conf.trim_init_x:conf.trim_end_x, 
                    conf.trim_init_y:conf.trim_end_y]
    else:
        trimmed_images = np.empty((
                                    len(img_array), 
                                    conf.trim_end_x-conf.trim_init_x, 
                                    conf.trim_end_y-conf.trim_init_y, 
                                    conf.n_channels
                                ),
                                dtype=conf.pixel_type)

        for i in range(len(img_array)):
            trimmed_images[i] = img_array[i][conf.trim_init_x:conf.trim_end_x, 
                    conf.trim_init_y:conf.trim_end_y]

        return trimmed_images   




#########################################################################
## ------------------------ Private functions ------------------------ ##
#########################################################################

# Per pixel kernel to convert an image to black and white
@cuda.jit
def __convert_to_bw_kernel(img_array, bw_array):
    # calculate the x and y indices of the current thread
    x, y = cuda.grid(2)

    # calculate the grayscale value of the pixel at (x, y)
    r, g, b = img_array[x, y, 0], img_array[x, y, 1], img_array[x, y, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    # convert to 8-bit integer
    gray = conf.pixel_type(gray/255)

    # set the pixel normalized value in the black and white image array
    bw_array[x, y, 0] = gray