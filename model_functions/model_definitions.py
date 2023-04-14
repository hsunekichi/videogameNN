import tensorflow as tf
from tensorflow import keras
import config as conf
import model_functions.model_execution as model_exec
from sklearn.decomposition import PCA   
from dataclasses import dataclass
import numpy as np


#########################################################
## ----------------- Data definition ----------------- ##
#########################################################

@dataclass
class training_options:
    PCA_model: PCA
    early_stopping_patience: int
    scale: int = conf.scale
    n_timesteps: int = conf.n_timesteps
    epoch: int = 200
    generator_batch_size: int = 32
    batch_size: int = 32
    validation_split: int = 0.1




#########################################################
## ------------------ Custom layers ------------------ ##
#########################################################

class CoordinateExpansion(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CoordinateExpansion, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.height = input_shape[1]
        self.width = input_shape[2]
        
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]

        # Creates coordinates
        i_coords = tf.linspace(start= 0.0, stop= 1, num= self.height)
        j_coords = tf.linspace(start= 0.0, stop= 1, num= self.width)
        
        # Reshapes coordinates 
        i_coords = tf.reshape(i_coords, (1, self.height, 1, 1))
        j_coords = tf.reshape(j_coords, (1, 1, self.width, 1))

        # Copies each coordinate dimension over the other one
        # (Gets the complete coordinate matrixes)
        i_coords = tf.tile(i_coords, (batch_size, 1, self.width, 1))
        j_coords = tf.tile(j_coords, (batch_size, self.height, 1, 1))

        # Adds the coordinates to the image
        # Either to the new dimension or to the channels
        expanded_inputs = tf.concat([inputs, i_coords, j_coords], axis=3)

        return expanded_inputs
    
    def compute_output_shape(self, input_shape):
        return input_shape[:3] + (input_shape[3] + 2,)
    
    def get_config(self):
        config = super(CoordinateExpansion, self).get_config()
        return config
    



# Define the FlattenChannels layer
class FlattenChannels(keras.layers.Layer):
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]
        channels = tf.shape(inputs)[3]
        # Reshape each channel independently
        outputs = []
        for i in range(channels):
            channel = inputs[..., i]
            flattened_channel = tf.reshape(channel, (batch_size, height * width))
            outputs.append(flattened_channel)
        # Stack the flattened channels along the last axis
        outputs = tf.stack(outputs, axis=-1)
        return outputs
    
    def compute_output_shape(self, input_shape):
        return input_shape[:3] + (input_shape[3],)
    
    def get_config(self):
        config = super(FlattenChannels, self).get_config()
        return config
    


class MatrixToImageLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MatrixToImageLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(MatrixToImageLayer, self).build(input_shape)

    def call(self, inputs):
        # Get the dimensions of the input tensor
        batch_size, height, width = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2]
        channels = tf.shape(inputs)[3]

        # Reshape the input tensor to convert it into a 2D image
        # The new shape will be (batch_size, height, width, channels)
        outputs = tf.reshape(inputs, (batch_size, height, width, channels))

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], input_shape[2], input_shape[3]
    


class ImageBuffer(keras.layers.Layer):
    def __init__(self, buffer_size, dtype='float32', **kwargs):
        self.buffer_size = buffer_size
        self._dtype=dtype

        super(ImageBuffer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.buffer_shape = input_shape.as_list()
        self.buffer_shape[0] = self.buffer_size
        
        self.buffer = keras.backend.zeros(self.buffer_shape)

        super(ImageBuffer, self).build(input_shape)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        output = tf.TensorArray(dtype=tf.float32, size=batch_size)
        
        for i in tf.range(batch_size):
            new_feature = inputs[i]

            # Updates the buffer with current feature
            # Adds the new feature to the buffer and removes the oldest one
            buffer = keras.backend.concatenate([self.buffer[1:], 
                                                keras.backend.expand_dims(new_feature, 0)
                                                ], 
                                                axis=0)
            self.add_update([(self.buffer, buffer)])

            # Concatenates all buffer features in a single array
            output = output.write(i, buffer)
            
        # Stack the output tensor array into a single tensor
        output = output.stack()

        return output

    def compute_output_shape(self):
        return self.buffer_shape


##########################################################
## ----------------- Model definition ----------------- ##
##########################################################

def create_model_temporal_XXS(shape, nOutputs):
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

    model = keras.models.Sequential([
        
        # Input layer
        #keras.layers.Reshape((img_heigth, img_width, 1), input_shape=(np.shape(dataset))),
        keras.layers.InputLayer(input_shape = shape),

        #   Convolutional layers
        keras.layers.TimeDistributed(
            keras.layers.Conv2D(32, (3, 3), activation='relu')),

        keras.layers.TimeDistributed(    
            keras.layers.MaxPool2D((3, 3))),


        keras.layers.TimeDistributed(    
            keras.layers.Conv2D(16, (2, 2), activation='relu')),

        keras.layers.TimeDistributed(    
            keras.layers.AvgPool2D((3, 3))),

        keras.layers.TimeDistributed(
            keras.layers.Flatten()),

        keras.layers.LSTM(32, activation='tanh'),
        
        #keras.layers.Dropout(0.1),

        #   Dense layers
        # Flatten the 2D feature maps into a 1D feature vector
        #keras.layers.TimeDistributed(
        #    keras.layers.Flatten()),


        keras.layers.Dense(16, activation='tanh'),

        #   Output layer
        keras.layers.Dense(nOutputs, activation='sigmoid')
    ])

    print(model.summary())

    model.compile(optimizer='nadam',
                loss= model_exec.focal_loss(),
                #metrics=[model_exec.focal_loss()]
                #loss="binary_crossentropy",
                metrics=["binary_crossentropy", tf.keras.metrics.Recall(thresholds=conf.threshold_output)])

    return model



def create_model_buffered_XXS(shape, nOutputs):

    #tf.keras.mixed_precision.set_global_policy('mixed_float16')    

    ############## Model layers ##############
    input = keras.layers.Input(shape=shape)     #, dtype='float16')

    compression_layer = keras.models.Sequential([  

        #CoordinateExpansion(),  # Adds two coordinate channels (i, j)

        # Reshape the input tensor to separate each channel into its own spatial dimension
        # The new shape will be (batch_size, height, width, img+coords, 1)
        #keras.layers.Reshape((input.shape[1], input.shape[2], 3, 1)),

        #   Convolutional layers
        keras.layers.Conv2D(8, (3, 3), activation='relu'),
        keras.layers.Conv2D(8, (3, 3), activation='relu'),
        keras.layers.MaxPool2D((3, 3)),

        keras.layers.Conv2D(16, (2, 2), activation='relu'),
        keras.layers.AvgPool2D((2, 2)),

        keras.layers.Conv2D(16, (2, 2), activation='relu'),
        keras.layers.AvgPool2D((2, 2)),

        keras.layers.Conv2D(32, (2, 2), activation='relu'),

        # Fuses 3 channels in one
        #MatrixToImageLayer(),
        #keras.layers.Conv2D(1, (1, 1), activation='relu')

        keras.layers.Flatten()
    ],
    name="Compression")

    #print(compression_layer(input).layer[-1].output_shape)
    
    lst_memory_layer = keras.models.Sequential([

        ImageBuffer(buffer_size=conf.n_timesteps, name="Memory_buffer"),
        keras.layers.LSTM(16, activation='tanh'),
        
        #keras.layers.Dropout(0.1),
    ],
    name="LST_memory")

    output_layer = keras.models.Sequential([
        keras.layers.Dense(32, activation='tanh'),
        keras.layers.Dense(16, activation='tanh'),

        #   Output layer
        keras.layers.Dense(nOutputs, activation='sigmoid')
    ],
    name="Dense_classifier")


    ############## Model ##############

    internal_compressed_input = compression_layer(input)
    internal_lst_memory = lst_memory_layer(internal_compressed_input)
    output = output_layer(internal_lst_memory)

    model = keras.Model(input, output)

    print(model.summary())

    model.compile(optimizer='nadam',
                #loss= model_exec.ChangeBinaryCrossentropy(),
                #metrics=[model_exec.focal_loss()]
                loss="binary_crossentropy",
                metrics=["binary_crossentropy", tf.keras.metrics.Recall(thresholds=conf.threshold_output)])

    return model



def create_model_temporal_XS(shape, nOutputs):
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

    model = keras.models.Sequential([
        
        # Input layer
        #keras.layers.Reshape((img_heigth, img_width, 1), input_shape=(np.shape(dataset))),
        keras.layers.InputLayer(input_shape = shape),

         #   Convolutional layers
        keras.layers.TimeDistributed(
            keras.layers.Conv2D(32, (3, 3), activation='relu')),

        keras.layers.TimeDistributed(    
            keras.layers.MaxPool2D((3, 3))),


        keras.layers.TimeDistributed(    
            keras.layers.Conv2D(16, (2, 2), activation='relu')),

        keras.layers.TimeDistributed(    
            keras.layers.AvgPool2D((3, 3))),

        keras.layers.TimeDistributed(
            keras.layers.Flatten()),


        keras.layers.TimeDistributed(
            keras.layers.Dense(32, activation='tanh')),

        #keras.layers.LSTM(32, activation='tanh', return_sequences = True),
        keras.layers.LSTM(16, activation='tanh'),

        keras.layers.Dropout(0.05),

        #keras.layers.Dense(32, activation='tanh'),
        keras.layers.Dense(16, activation='tanh'),

        #   Output layer
        keras.layers.Dense(nOutputs, activation='sigmoid')
    ])

    print(model.summary())

    model.compile(optimizer='adam',
                loss= model_exec.focal_loss(),
                metrics=[model_exec.focal_loss()])

    return model



def create_model_temporal_XL(shape, nOutputs):
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print(shape)
    model = keras.models.Sequential([
        
        # Input layer
        #keras.layers.Reshape((img_heigth, img_width, 1), input_shape=(np.shape(dataset))),
        keras.layers.InputLayer(input_shape = shape, dtype = 'float16'),

        #   Convolutional layers
        keras.layers.TimeDistributed(
            keras.layers.Conv2D(64, (3, 3), activation='relu', dtype = 'float16')),

        keras.layers.TimeDistributed(    
            keras.layers.MaxPooling2D((3, 3), dtype = 'float16')),


        keras.layers.TimeDistributed(    
            keras.layers.Conv2D(64, (2, 2), activation='relu')),

        keras.layers.TimeDistributed(    
            keras.layers.Conv2D(16, (2, 2), activation='relu')),

        keras.layers.TimeDistributed(
            keras.layers.Flatten()),

        keras.layers.LSTM(64, activation='tanh'),
        
        keras.layers.Dropout(0.1),

        #   Dense layers
        # Flatten the 2D feature maps into a 1D feature vector
        #keras.layers.TimeDistributed(
        #    keras.layers.Flatten()),


        keras.layers.Dense(32, activation='tanh'),
        keras.layers.Dense(32, activation='tanh'),

        keras.layers.Dense(16, activation='tanh'),

        #   Output layer
        keras.layers.Dense(nOutputs, activation='sigmoid', dtype = 'float16')
    ])

    print(model.summary())

    model.compile(optimizer='adam',
                loss= model_exec.focal_loss(),
                metrics=[model_exec.focal_loss()])

    return model

## ----------------- Model definition ----------------- ##
def create_model_static(shape, nOutputs):
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

    model = keras.models.Sequential([
        
        # Input layer
        #keras.layers.Reshape((img_heigth, img_width, 1), input_shape=(np.shape(dataset))),
        keras.layers.InputLayer(input_shape = shape),

        #   Convolutional layers
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),

        keras.layers.Conv2D(64, (3, 3), activation='relu'),

        keras.layers.Conv2D(32, (3, 3), activation='relu'),
        keras.layers.AvgPool2D((2, 2)),

        keras.layers.Conv2D(32, (3, 3), activation='relu'),
        keras.layers.AvgPool2D((2, 2)),

        keras.layers.Flatten(),
        #   Dense layers
        keras.layers.Dense(64, activation='tanh'),

        keras.layers.Dense(32, activation='tanh'),
        keras.layers.Dropout(0.2),

        keras.layers.Dense(16, activation='linear'),

        #   Output layer
        keras.layers.Dense(nOutputs, activation='sigmoid')
    ])

    #print(model.summary())


    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['binary_crossentropy'])

    return model


def encoder_xl(shape, output_image_side_size):
    print("Shape: ", shape)
    print("output size: ", output_image_side_size)
    encoder = keras.models.Sequential([
        keras.layers.InputLayer(input_shape = shape),

        #keras.layers.Conv2D(64, (3, 3), activation='relu'),

        #keras.layers.Conv2D(32, (3, 3), activation='relu'),
        #keras.layers.MaxPool2D((2, 2)),

        #keras.layers.Conv2D(32, (3, 3), activation='relu'),

        keras.layers.Flatten(),
        keras.layers.Dense(512),       
        keras.layers.LeakyReLU(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(256),
        keras.layers.LeakyReLU(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(128),
        keras.layers.LeakyReLU(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(64),
        keras.layers.LeakyReLU(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(output_image_side_size*output_image_side_size),
        keras.layers.Reshape((output_image_side_size, output_image_side_size, 1))
    ])

    return encoder


def encoder_xs(shape, output_image_side_size):
    print("Shape: ", shape)
    print("output size: ", output_image_side_size)
    encoder = keras.models.Sequential([
        keras.layers.InputLayer(input_shape = shape),

        keras.layers.Flatten(),
        keras.layers.Dense(output_image_side_size*output_image_side_size),
        keras.layers.Reshape((output_image_side_size, output_image_side_size, 1))
    ])

    return encoder



def decoder_xl(shape, output_image_side_size):
    decoder = keras.models.Sequential([
        keras.layers.Flatten(input_shape = (output_image_side_size, output_image_side_size, 1)),
        keras.layers.Dense(64),
        keras.layers.LeakyReLU(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(128),
        keras.layers.LeakyReLU(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(256),
        keras.layers.LeakyReLU(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(512),
        keras.layers.LeakyReLU(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(shape[0]*shape[1]*shape[2]),
        keras.layers.Activation("sigmoid"),
        keras.layers.Reshape(shape)
    ])

    return decoder



def decoder_xs(shape, output_image_side_size):
    decoder = keras.models.Sequential([
        keras.layers.Flatten(input_shape = (output_image_side_size, output_image_side_size, 1)),
        keras.layers.Dense(shape[0]*shape[1]*shape[2]),
        keras.layers.Activation("sigmoid"),
        keras.layers.Reshape(shape)
    ])

    return decoder