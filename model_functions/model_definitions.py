import tensorflow as tf
from tensorflow import keras
import config as conf
import model_functions.model_execution as model_exec
from sklearn.decomposition import PCA   
from dataclasses import dataclass
import numpy as np
import cv2


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



class KMeansClustering(keras.layers.Layer):
    def __init__(self, num_clusters, **kwargs):
        super(KMeansClustering, self).__init__(**kwargs)
        self.num_clusters = num_clusters
    
    def build(self, input_shape):
        self.centroids = self.add_weight(
            name="centroids",
            shape=(self.num_clusters, input_shape[-1]),
            initializer="glorot_uniform",
            trainable=True,
        )
        super(KMeansClustering, self).build(input_shape)
    
    def call(self, inputs):
        # Flatten the input tensor
        inputs_flat = tf.reshape(inputs, (-1, inputs.shape[-1]))
        
        # Compute the distances between the inputs and centroids
        distances = tf.reduce_sum(tf.square(tf.expand_dims(inputs_flat, axis=1) - self.centroids), axis=-1)
        
        # Assign each input to the nearest centroid
        cluster_indices = tf.argmin(distances, axis=-1)
        clusters = tf.one_hot(cluster_indices, depth=self.num_clusters)
        
        # Update the centroids based on the assigned inputs
        counts = tf.reduce_sum(clusters, axis=0)
        new_centroids = tf.reduce_sum(tf.expand_dims(inputs_flat, axis=1) * tf.expand_dims(clusters, axis=-1), axis=0) / tf.expand_dims(counts, axis=-1)
        
        # Update the centroids using exponential moving average
        self.add_update(tf.compat.v1.assign(self.centroids, 0.9 * self.centroids + 0.1 * new_centroids))
        
        # Reshape the clusters tensor to match the shape of the input tensor
        clusters = tf.reshape(clusters, tf.shape(inputs)[:-1] + (self.num_clusters,))
        
        return clusters
    
    def get_config(self):
        config = super(KMeansClustering, self).get_config()
        config.update({"num_clusters": self.num_clusters})
        return config



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


def create_model_buffered_XXS(shape, nOutputs):

    #tf.keras.mixed_precision.set_global_policy('mixed_float16')    

    ############## Model layers ##############
    input = keras.layers.Input(shape=shape)     #, dtype='float16')


    ###################### Compression layer ######################
        
    compression_layer = keras.models.Sequential([  

        #CoordinateExpansion(),  # Adds two coordinate channels (i, j)

        # Reshape the input tensor to separate each channel into its own spatial dimension
        # The new shape will be (batch_size, height, width, img+coords, 1)
        #keras.layers.Reshape((input.shape[1], input.shape[2], 3, 1)),

        #   Convolutional layers
        #keras.layers.Conv2D(8, (3, 3), activation='relu'),
        keras.layers.MaxPool2D((3, 3)),

        keras.layers.Conv2D(8, (3, 3), activation='relu'),
        keras.layers.MaxPool2D((3, 3)),

        keras.layers.Conv2D(16, (2, 2), activation='relu'),
        keras.layers.AvgPool2D((2, 2)),
        
        keras.layers.Conv2D(32, (2, 2), activation='relu'),
        keras.layers.AvgPool2D((2, 2)),

        # Fuses 3 channels in one
        #MatrixToImageLayer(),
        #keras.layers.Conv2D(1, (1, 1), activation='relu')

        keras.layers.Flatten(),
    ],
    name="Compression")
    

    # Inicial, segundo, maxPool3 y 1 2d, 2 capas (15)
    ###################### Attention layer ######################

    def attention_layer(input):

        #input1 = keras.layers.Dense(64, activation='tanh')(input)

        internal_compressed_input = keras.backend.expand_dims(input, axis=-1)

        attention_layer_internal = keras.layers.MultiHeadAttention(num_heads=16, key_dim=16)
        #attention_layer_internal = keras.layers.AdditiveAttention(use_scale=True)

        #internal_attention = attention_layer_internal([internal_compressed_input, 
        #                                    internal_compressed_input])
        
        internal_attention = attention_layer_internal(internal_compressed_input,
                                                        internal_compressed_input)

        return keras.backend.squeeze(internal_attention, axis=-1)


    class show_layer(keras.layers.Layer):
        def __init__(self, **kwargs):
            super(show_layer, self).__init__(**kwargs)
        
        def build(self, input_shape):
            super(show_layer, self).build(input_shape)
            
        def call(self, inputs):
            print(np.shape(inputs))
            image = tf.make_ndarray(inputs[0, :, :, 0])
            cv2.imshow("first_layer", image)
            cv2.waitKey(0)

            return inputs


    ###################### Memory layer ######################

    lst_memory_layer = keras.models.Sequential([

        ImageBuffer(buffer_size=conf.n_timesteps, name="Memory_buffer"),
        keras.layers.LSTM(16, activation='tanh'),
        
        #keras.layers.Dropout(0.1),
    ],
    name="LST_memory")


    ###################### Output layer ######################

    output_layer = keras.models.Sequential([
        
        #keras.layers.Dense(64, activation='tanh'),
        keras.layers.Dense(32, activation='tanh'),
        keras.layers.Dense(16, activation='tanh'),

        #   Output layer
        keras.layers.Dense(nOutputs, activation='sigmoid')
    ],
    name="Dense_classifier")


    ############## Model ##############

    first_layer = keras.layers.Conv2D(8, (3, 3), activation='relu')(input)
    showed_layer = show_layer()(first_layer)


    internal_compressed_input = compression_layer(showed_layer)
    internal_attention = attention_layer(internal_compressed_input)
    #internal_lst_memory = lst_memory_layer(internal_compressed_input)

    #classifier_input = keras.layers.concatenate([internal_compressed_input, internal_lst_memory])
    output = output_layer(internal_attention)

    model = keras.Model(input, output)

    print(model.summary())

    model.compile(optimizer='nadam',
                loss= model_exec.ChangeBinaryCrossentropy(),
                #metrics=[model_exec.focal_loss()]
                #loss="binary_crossentropy",
                metrics=["binary_crossentropy", tf.keras.metrics.Recall(thresholds=conf.threshold_output)])

    return model


## ----------------- Model definition ----------------- ##

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