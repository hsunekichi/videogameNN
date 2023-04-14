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



class ChangeBinaryCrossentropy(keras.losses.BinaryCrossentropy):
    def __init__(self, change_factor=1, prev_output=None, **kwargs):
        super(ChangeBinaryCrossentropy, self).__init__(**kwargs)
        self.prev_output = prev_output
        self.change_factor = change_factor
        
    def call(self, y_true, y_pred):

        loss_value = super(ChangeBinaryCrossentropy, self).call(y_true, y_pred)

        if self.prev_output != y_true:
            loss_value *= self.change_factor

        return loss_value



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


