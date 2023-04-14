import tensorflow as tf
from tensorflow import keras
import numpy as np
from data_processing import data_generation as data_gen
import config as conf
from dataclasses import dataclass
import glob
from sklearn.decomposition import PCA
import model_functions.model_definitions as model_def
import cv2
import inspect
import random


# Entrena al modelo sobre un solo video
def train_one_video(Xtr, ytr, model, options: model_def.training_options):

    
    ## ----------------- Data preparation ----------------- ##


    ## ----------------- Training ----------------- ##

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=options.early_stopping_patience, restore_best_weights=True)

    model.fit(Xtr, ytr, epochs=options.epoch, 
                callbacks=[callback], batch_size=options.batch_size, validation_split=0.1)     #, validation_data=val_generator)

# Entrena al modelo con una colección completa de vídeos
def train_model(vid_path, input_path, test_vid_path, test_log_path, crear_modelo, options: model_def.training_options):

    # Gets all the files that match the regular expression
    videos = glob.glob(vid_path)
    inputs = glob.glob(input_path)

    # Sorts the files by name
    videos.sort()
    inputs.sort()

    ## ----------------- Testing ----------------- ##

    #Xtest = data_gen.load_all_frames(test_vid_path, scale=options.scale, pca=options.PCA_model)
    #ytest = data_gen.load_all_inputs(test_log_path)
    #Xtest_window, ytest_window = data_gen.sliding_window(Xtest, ytest, window_size=options.n_timesteps)


    ## ----------------- Training ----------------- ##
    """
    for i in range(len(videos)):
        print("\nEntrenando con ", videos[i], "\n")

        Xtr = data_gen.load_one_video_frames(videos[i], scale=options.scale, pca=options.PCA_model)
        ytr = data_gen.load_one_file_inputs(inputs[i])

        # Caso base, crea el modelo
        if i == 0:
            shape = (options.n_timesteps, np.shape(Xtr)[1], np.shape(Xtr)[2], np.shape(Xtr)[3])
            model = crear_modelo(shape, conf.n_outputs)

        train_one_video(Xtr, ytr, model, options) 
    """        

    Xtr = data_gen.load_all_frames(vid_path, scale=options.scale, pca=options.PCA_model)
    ytr = data_gen.load_all_inputs(input_path)

    if inspect.isfunction(crear_modelo):
        # Caso base, crea el modelo
        print("Creando modelo")
        model = crear_modelo(np.shape(Xtr[0]), conf.n_outputs)
    else:
        # El modelo ya existe
        model = crear_modelo


    train_one_video(Xtr, ytr, model, options) 

    return model#, Xtest_window, ytest_window



# Rentrena al modelo con una colección nueva de vídeos
def retrain_existant_model(vid_path, input_path, model, options: model_def.training_options):

    # Gets all the files that match the regular expression
    videos = glob.glob(vid_path)
    inputs = glob.glob(input_path)

    # Sorts the files by name
    videos.sort()
    inputs.sort()

    ## ----------------- Training ----------------- ##

    for i in range(len(videos)):
        print("\nEntrenando con ", videos[i], "\n")

        Xtr = data_gen.load_one_video_frames(videos[i], scale=options.scale, pca=options.PCA_model)
        ytr = data_gen.load_one_file_inputs(inputs[i])

        train_one_video(Xtr, ytr, model, options) 



# Entrena varios modelos cambiando sus características
def train_multiple_models(vid_path, input_path, test_vid_path, test_log_path, model_list, options_list):

    models = []
    histories = []

    for i in range(len(model_list)):

        # Trains the model
        #model, Xtest, ytest = train_model(vid_path, input_path, test_vid_path, test_log_path, model_list[i], options_list[i])
        model = train_model(vid_path, input_path, test_vid_path, test_log_path, model_list[i], options_list[i])

        # Evaluates the model
        #history = model.evaluate(Xtest, ytest)
        history = []
        
        # Stores the results
        models.append(model)
        histories.append(history)

    return models, histories


def train_autoencoder(vid_path):
    side = conf.PCA_image_side
    Xtr = data_gen.load_all_frames(vid_path)  # Loads tr frames

    #print(np.shape(Xtr[100]))
    #cv2.imshow("Original", Xtr[100])
    #cv2.waitKey(10)
    img_shape = np.shape(Xtr)[1:]   # Gets the shape of each of the frames


    model = create_autoencoder_model(img_shape, side)

    # Entrena y almacena el modelo PCA
    print()
    print("Entrenando autoencoder...")

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)


    model.fit(Xtr, Xtr, epochs=100, batch_size=32, 
                callbacks=[callback], shuffle=True, validation_split = 0.1)

    
    #cv2.imshow("decoded", model.predict(Xtr)[100])
    #cv2.waitKey(0)

    return model
    


# Crea un modelo PCA y lo entrena
def create_autoencoder_model(img_shape, side):

    #PCA_model = PCA(n_components=size)

    # Carga los datos de entrenamiento y test del PCA


    #kpca = KernelPCA(n_components=300, kernel="rbf", gamma=0.1, fit_inverse_transform=True)

    # Convierte los frames en arrays de pixeles 2D
    #Xtr_array = np.reshape(Xtr, (np.shape(Xtr)[0], np.shape(Xtr)[1]*np.shape(Xtr)[2]))

    encoder = model_def.encoder_xl(img_shape, side)
    decoder = model_def.decoder_xl(img_shape, side)

    img = keras.layers.Input(img_shape)
    latent_vector = encoder(img)
    output = decoder(latent_vector)

    model = keras.models.Model(inputs = img, outputs = output)
    #model.compile("nadam", loss = "mean_absolute_error")
    model.compile("nadam", loss = "binary_crossentropy")


    return model