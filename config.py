import numpy as np
import math

maxThreadsPerBlock = 1024
dimmSize = int(math.sqrt(maxThreadsPerBlock))

trim_init_x = 487
trim_end_x = 587

trim_init_y = 534
trim_end_y = 634

trim_size_x = (trim_end_x - trim_init_x)
trim_size_y = (trim_end_y - trim_init_y)

screen_height = 660
screen_width = 1136
record_fps = 30

n_timesteps = 5
n_channels = 1
n_outputs = 1
n_bidirectional_axis = 0

## ------- PCA ------- ##
downscale_batch_size = 2000
PCA_image_side = 100


scale = 3

pixel_type = np.float16
block_dim = (dimmSize, dimmSize)

threshold_input = 0.15    # Threshold to discretize the controller input values
threshold_output = 0.5    # Threshold to classify model outputs as 1 or 0