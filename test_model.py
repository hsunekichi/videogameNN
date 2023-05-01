from tensorflow import keras
import tensorflow as tf
import numpy as np
import cv2
from numba import cuda
import data_processing.image_preproc as image_preproc
import data_processing.data_generation as data_generation
import model_functions.model_execution as model_exec
import data_processing.image_postproc as image_postproc
import model_functions.model_definitions as models
import config as conf
import sys
import time
#from skcuda.linalg import PCA
from sklearn.decomposition import PCA
import pickle as pk

def PCA_np(X , n_components):
     
    #Step-1
    X_meaned = X - np.mean(X , axis = 0)
     
    #Step-2
    cov_mat = np.cov(X_meaned , rowvar = False)
     
    #Step-3
    eigen_values , eigen_vectors = np.linalg.eigh(cov_mat)
     
    #Step-4
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:,sorted_index]
     
    #Step-5
    eigenvector_subset = sorted_eigenvectors[:,0:n_components]
     
    #Step-6
    X_reduced = np.dot(eigenvector_subset.transpose() , X_meaned.transpose() ).transpose()
     
    return X_reduced

## ----------------- Preparation of the data ----------------- ##

nombre = sys.argv[1]
vid_path = "datasets/"+nombre+"/videos/vid*.mp4"    # Directory with the frames
input_path = "datasets/"+nombre+"/logs/log*.txt"    # File with the inputs

#XrealData = processData.load_real_frames(vid_path)
Xdata = data_generation.load_all_frames(vid_path)  # Loads the normalized pixels
ydata = data_generation.load_all_inputs(input_path)

data_size = len(ydata)
x_size = len(Xdata)
print ("y size: ", data_size)
print ("X size: ", x_size)

show_images = image_postproc.regenerate_images(Xdata)

#Xdata = Xdata[(x_size-data_size):] # Adjusts the size of the inputs (erases inital frames)
#Xreal = XrealData[(x_size-data_size):] # Adjusts the size of the inputs (erases inital frames)

#pca = pk.load(open("pca.pkl",'rb'))
pca = None


## ----------------- Evaluation ----------------- ##

ruta_modelo = "trained_models/modelos_"+nombre+"/modelo_"+sys.argv[2]+"_completo.h5"
model = keras.models.load_model(ruta_modelo, custom_objects={'loss': model_exec.ChangeBinaryCrossentropy(),
                                                             'ImageBuffer': models.ImageBuffer})

#print(modelFunc.porcentaje_acierto_secuencia(Xgenerated, Ygenerated, model))

predY = model_exec.predict(model, Xdata)

for i in range(len(predY)):
    print(i, ": ", ydata[i], " - ", predY[i])

cursor = 200

while True:
    print("Predicted - Real: ", predY[cursor], " - ", ydata[cursor])
    #for i in range(conf.n_timesteps):
    #    cv2.imshow("Paquete "+str(cursor)+" frame "+str(cursor+i), image_postproc.regenerate_images(np.array([Xtest_window[init][i]]))[0])

    cv2.imshow("Frame: "+str(cursor), show_images[cursor])

    # Gets a character from the keyboard
    key = cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()

    # If the character is 'd', the cursor moves to the right
    if key == ord("d"):
        if cursor < len(predY):
            cursor += 1
    
    # If the character is 'a', the cursor moves to the left
    if key == ord("a"):
        if cursor > 0:
            cursor -= 1

    if key == ord("c"):
        if cursor < len(predY):
            cursor += 10

    if key == ord("z"):
        if cursor < len(predY):
            cursor -= 10

    if key == ord("q"):
        break


