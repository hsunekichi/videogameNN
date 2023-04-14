import model_functions.model_definitions as models
import data_processing.data_generation as data_gen
import data_processing.image_postproc as post_proc
import model_functions.model_training as model_train
import model_functions.model_execution as model_exec
from tensorflow import keras
import config as conf
import sys
import os
import glob
import numpy as np
import pickle as pk
import cv2

## ----------------- Preparation of the data ----------------- ##

nombre = sys.argv[1]
if len(sys.argv) > 2:
    id = sys.argv[2]
else:
    id = None

vid_path = "datasets/"+nombre+"/videos/vid*.mp4"    # Directory with the frames
input_path = "datasets/"+nombre+"/logs/log*.txt"    # File with the inputs

X_test_path = "datasets/"+nombre+"/videos/vid*.mp4"    
y_test_path = "datasets/"+nombre+"/logs/log*.txt"   

# Si no existe un directorio para los modelos, lo crea
if not os.path.exists("trained_models/modelos_"+nombre):
    os.makedirs("trained_models/modelos_"+nombre)


"""
if not os.path.exists("trained_models/modelos_"+nombre+"/autoencoder.pkl"):

    PCA_model = model_train.train_autoencoder(vid_path)
    pk.dump(PCA_model, open("trained_models/modelos_"+nombre+"/autoencoder.pkl","wb"))
    
    Xtest = data_gen.load_all_frames(X_test_path, conf.scale, PCA_model)  # Loads test frames
    Xtest_real = data_gen.load_all_frames(X_test_path, conf.scale, None)  # Loads test frames

    regen_test = post_proc.regenerate_images(Xtest)
    regen_real = post_proc.regenerate_images(Xtest_real)

    cv2.imshow("Frame 200 decoded: ", regen_test[200])
    cv2.imshow("Frame 200 real: ", regen_real[200])
    cv2.waitKey(0)

else:
    # Carga un modelo PCA ya existente
    PCA_model = pk.load(open("trained_models/modelos_"+nombre+"/autoencoder.pkl",'rb'))
    print("Modelo PCA cargado")
"""

PCA_model = None


if id != None:
    model = keras.models.load_model("trained_models/modelos_"+nombre+"/modelo_"+id+".h5", 
                                    custom_objects={'loss': model_exec.focal_loss(),
                                                    'ImageBuffer': models.ImageBuffer})
    model_list = [model]
else:
    model_list = [models.create_model_buffered_XXS]



options_list = [
    models.training_options(
        n_timesteps=conf.n_timesteps, 
        PCA_model = PCA_model,
        early_stopping_patience = 10,
        scale = conf.scale,
        epoch = 150,
        generator_batch_size = 1024,
        batch_size = 32,
        validation_split = 0.1)
]

modelos, historias = model_train.train_multiple_models(vid_path, input_path, X_test_path, y_test_path, model_list, options_list)

## ----------------- Testing ----------------- ##

print(historias)

# Especifica el patrón de nombres de archivo que quieres buscar
pattern = "trained_models/modelos_"+nombre+"/modelo_*"

# Usa glob para obtener una lista de nombres de archivo que cumplan el patrón
files = glob.glob(pattern)

# Obtener el ID más alto
max_id = -1
max_id_file = ""
for file in files:
    number = file.split("modelo_")[1].split(".h5")[0]
        
    file_id = int(number)
    if file_id > max_id:
        max_id = file_id
        max_id_file = file

indice_modelo = max_id+1

i=0
for modelo in modelos:
    print("Saving model "+str(indice_modelo)+"...")
    ruta = "trained_models/modelos_"+nombre+"/modelo_"+str(indice_modelo)+".h5"
    modelo.save(ruta)

    #ruta = "trained_models/modelos_"+nombre+"/historia_"+str(indice_modelo)+".txt"
    #file = open(ruta,"w")

    #file.write(str(historias[i][0]))

    indice_modelo+=1
    i+=1









