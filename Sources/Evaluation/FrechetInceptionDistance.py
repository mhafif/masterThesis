# example of calculating the frechet inception distance in Keras for cifar10
import numpy as np
import os
import glob
from numpy import cov, trace, iscomplexobj, asarray
from numpy.random import shuffle
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from PIL import Image
from MnistClassifier import buildModel, preprocess_mnist

# calculate frechet inception distance
def compute_fid(model, images1, images2):
    # calculate activations
    act1 = model.predict(images1)
    act2 = model.predict(images2)
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

if __name__ == "__main__":
    # Current Folder
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    # Folder List of images
    foloderList = ["mnistGAN","mnistVAE","mnistVAEGAN","celebaGAN","celebaVAE","celebaVAEGAN"]
    
    # Folder of images
    realFolderCeleba = "celebaReal"
    realFolderMnist = "mnistReal"
    # Load images
    pathRealFolderCeleba = os.path.join(CURRENT_DIR, realFolderCeleba, "*.jpg")
    imagesFilesRealCeleba = glob.glob(pathRealFolderCeleba)
    pathRealFolderMnist = os.path.join(CURRENT_DIR, realFolderMnist, "*.jpg")
    imagesFilesRealMnist = glob.glob(pathRealFolderMnist)

    for foldername in foloderList:
        pathGenFolder = os.path.join(CURRENT_DIR, foldername, "*.jpg")
        imagesFilesGen = glob.glob(pathGenFolder)
        if "mnist" in foldername:
            # convert to array of images
            images_real = np.array([np.array((Image.open(fname))) for fname in imagesFilesRealMnist])
            images_gen = np.array([np.array((Image.open(fname))) for fname in imagesFilesGen])
            images_gen = images_gen[:,:,:,0]
            processed_real = preprocess_mnist(images_real)
            processed_gen = preprocess_mnist(images_gen)

            # prepare the mnist classifier model
            model, model_activation = buildModel()
            model.load_weights(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mnistWeightsClassifier.h5"))

            # compute FID
            fid = compute_fid(model_activation, processed_real, processed_gen)

        else:
            # convert to array of images and resize to 299x299
            images_real = np.array([np.array((Image.open(fname)).resize((299,299))) for fname in imagesFilesRealCeleba])
            images_gen = np.array([np.array((Image.open(fname)).resize((299,299))) for fname in imagesFilesGen])
            processed_real = preprocess_input(images_real)
            processed_gen = preprocess_input(images_gen)

            # prepare the inception v3 model
            model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))

            # compute inception score
            fid = compute_fid(model, processed_real, processed_gen)
        
        print('FID Score for {} folder is: %.3f'.format(foldername) % fid)