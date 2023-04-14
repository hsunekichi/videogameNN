from sklearn.decomposition import PCA
import cv2
import numpy as np
import config as conf



def show_PCA_images(Xtr, Xtest, PCA_model):
    
    PCA_size = conf.PCA_image_side*conf.PCA_image_side

    # Reshape to 2D
    reduced_test = np.reshape(Xtest, (np.shape(Xtest)[0], PCA_size))
    reduced_tr = np.reshape(Xtr, (np.shape(Xtr)[0], PCA_size))

    # Get original images from PCA
    reformed_test = PCA_model.inverse_transform(reduced_test)
    reformed_tr = PCA_model.inverse_transform(reduced_tr)

    # Reshape to 3D
    reformed_test = np.reshape(reformed_test, (np.shape(reformed_test)[0], int(conf.screen_height/conf.scale), int(conf.screen_width/conf.scale), 1))
    reformed_tr = np.reshape(reformed_tr, (np.shape(reformed_tr)[0], int(conf.screen_height/conf.scale), int(conf.screen_width/conf.scale), 1))

    # Denormalize images
    Xtest_regenerated = regenerate_images(reformed_test)
    Xtr_regenerated = regenerate_images(reformed_tr)
    
    # Show images
    cv2.imshow("Compressed tr frame", Xtr_regenerated[0])
    cv2.imshow("Compressed test frame", Xtest_regenerated[0])

    cv2.waitKey(0)
    cv2.destroyAllWindows()

"""
def show_images():
    while True:
        init = cursor
        print("Predicted - Real: ", predY[init], " - ", Ytest_window[init])
        for i in range(conf.n_timesteps):
            cv2.imshow("Paquete "+str(cursor)+" frame "+str(cursor+i), image_postproc.regenerate_images(np.array([Xtest_window[init][i]]))[0])

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

        if key == ord("q"):
            break
"""




def regenerate_images(img_array):

    # Denormalizes the pixels in the image and makes them integers
    denormalized_images = img_array * 255
    denormalized_images = denormalized_images.astype(np.uint8)

    if np.shape(denormalized_images)[len(np.shape(denormalized_images))-1] == 1:
        # Adds the three channels to the image
        denormalized_images = np.concatenate((denormalized_images, denormalized_images, denormalized_images), axis=len(np.shape(denormalized_images))-1)

    return denormalized_images
    