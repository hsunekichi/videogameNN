import tensorflow as tf
from tensorflow import keras
import numpy as np
import config as conf


#########################################################
## -------------------- Prediction ------------------- ##
#########################################################

def predict(model, X):

    single_image = False
    # If the input is a single image, it is converted to a batch of 1 image
    if len(np.shape(X)) == 3:
        single_image = True
        X = np.expand_dims(X, axis=0)

    predY = model.predict(X)

    binaryPredictions = (predY > conf.threshold_output).astype(int)

    if single_image:
        return binaryPredictions[0]
    else:
        return binaryPredictions





######################################################
## ------------------ Evaluation ------------------ ##
######################################################

# Focal loss implementation
def focal_loss(gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -tf.reduce_mean(alpha * tf.pow(1. - pt_1, gamma) * tf.math.log(pt_1 + keras.backend.epsilon()) + (1 - alpha) * tf.pow(pt_0, gamma) * tf.math.log(1. - pt_0 + keras.backend.epsilon()))
    return loss


def ChangeBinaryCrossentropy(n=2):
    """
    Custom binary cross entropy loss function that multiplies the loss by a factor 'n'
    if the real output is different from the previous one.
    
    Args:
    n (float): Multiplication factor for the loss if the real output is different from the previous one.

    Returns:
    loss (function): Custom loss function.
    """
    def loss(y_true, y_pred):
        # Get the previous output from the previous batch
        prev_output = keras.backend.concatenate([keras.backend.zeros_like(y_true[0:1]), y_true[:-1]], axis=0)

        # Compute the binary cross entropy loss
        binary_crossentropy = keras.backend.binary_crossentropy(y_true, y_pred)

        # Compute the difference between the current output and the previous output
        output_difference = keras.backend.abs(y_true - prev_output)

        # Multiply the binary cross entropy loss by 'n' if the outputs are different
        weighted_binary_crossentropy = keras.backend.switch(keras.backend.not_equal(output_difference, 0), n * binary_crossentropy, binary_crossentropy)

        return keras.backend.mean(weighted_binary_crossentropy)

    return loss


# Percentaje of completely well predicted outputs
def tasa_acierto(X, Y, model):
    predY = predict(model, X)

    # Calculates the accuracy
    aciertos = 0
    for i in range(len(Y)):
        if np.array_equal(Y[i], predY[i]):
            aciertos += 1

    return aciertos / len(Y)



# Average of the percentaje of well predicted outputs on each input
def porcentaje_acierto(X, Y, model):
    predY = predict(model, X)

    # Calculates the accuracy
    suma_porcentajes = 0
    for i in range(len(Y)):
        suma_porcentajes += np.sum(Y[i] == predY[i]) / len(Y[i])

    return suma_porcentajes / len(Y) * 100



# Average of the percentaje of well predicted outputs on each input
def porcentaje_acierto_secuencia(X, Y, model, batch_size):

    predY = predict(model, X, batch_size)
    
    # Calculates the accuracy
    suma_porcentajes = 0
    for i in range(len(Y)):
        suma_porcentajes += np.sum(Y[i] == predY[i][conf.n_timesteps-1]) / np.size(Y[i])

    return suma_porcentajes / np.size(Y) * 100


