# calculate inception score with Keras
import os
import glob
import numpy as np
from numpy import ones, expand_dims, log, mean, std, exp
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from math import floor
from PIL import Image
from MnistClassifier import buildModel
from MnistClassifier import preprocess_mnist

# assumes images have the shape 28x28 (mnist model) or 299x299 (inception model), pixels in [0,255]
def compute_inception_score(images, preprocess_function, type, n_split=10, eps=1E-16):
    if type == "celeba":
        # load inception v3 model
        model = InceptionV3()
    else:
        # load mnist model
        model, model_activation = buildModel()
        model.load_weights(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mnistWeightsClassifier.h5"))
    # convert from uint8 to float32
    processed = images.astype('float32')
    # pre-process raw images for mnist classifier model
    processed = preprocess_function(processed)
    # predict class probabilities for images
    yhat = model.predict(processed)
    # enumerate splits of images/predictions
    scores = list()
    n_part = floor(images.shape[0] / n_split)
    for i in range(n_split):
        # retrieve p(y|x)
        ix_start, ix_end = i * n_part, i * n_part + n_part
        p_yx = yhat[ix_start:ix_end]
        # calculate p(y)
        p_y = expand_dims(p_yx.mean(axis=0), 0)
        # calculate KL divergence using log probabilities
        kl_d = p_yx * (log(p_yx + eps) - log(p_y + eps))
        # sum over classes
        sum_kl_d = kl_d.sum(axis=1)
        # average over images
        avg_kl_d = mean(sum_kl_d)
        # undo the log
        is_score = exp(avg_kl_d)
        # store
        scores.append(is_score)
    # average across images
    is_avg, is_std = mean(scores), std(scores)
    return is_avg, is_std

if __name__ == "__main__":
    # Current Folder
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    # Folder List of images
    foloderList = ["mnistReal","mnistGAN","mnistVAE","mnistVAEGAN","celebaReal","celebaGAN","celebaVAE","celebaVAEGAN"]

    # Loop over the folder and compute IS
    for foldername in foloderList:
        # Load images
        pathFolderImages = os.path.join(CURRENT_DIR, foldername, "*.jpg")
        images_files = glob.glob(pathFolderImages)
        if "mnist" in foldername:
            # convert to array of images
            images = np.array([np.array((Image.open(fname))) for fname in images_files])
            if foldername != "mnistReal":
                images = images[:,:,:,0]
            # compute inception score
            is_avg, is_std = compute_inception_score(images, preprocess_mnist, "mnist")
        else:
            # convert to array of images and resize to 299x299
            images = np.array([np.array((Image.open(fname)).resize((299,299))) for fname in images_files])
            # compute inception score
            is_avg, is_std = compute_inception_score(images, preprocess_input, "celeba")
        
        print('Inception Score for {} folder is mu: {}, sigma: {}'.format(foldername, is_avg, is_std))