import keras
from keras.layers import Dropout, BatchNormalization, LeakyReLU, ReLU
from keras.preprocessing.image import img_to_array
import zipfile
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def batchnorm_activation_dropout(x, batchNorm=True, activation=None, dropout=False, dropRate=0.2):
    if batchNorm:
        x = BatchNormalization()(x)
    if activation == "leaky":
        x = LeakyReLU(alpha=0.2)(x)
    if activation == "relu":
        x = ReLU()(x)
    if dropout:
        x = Dropout(dropRate)(x)
    return x

def normalInit(stdv):
    return keras.initializers.RandomNormal(stddev=stdv)

def get_celeba_dataset(size_training, dataset_size, celeba_path):
    if dataset_size > 202599:
        raise ValueError('Celeba dataset has only 202599 images')
    if size_training > dataset_size:
        raise ValueError('Not enough images available for a training set of size {}'.format(size_training)) 
    images_numbers = np.arange(1,dataset_size+1)
    np.random.shuffle(images_numbers)
    images_numbers = images_numbers[:size_training]
    images_dataset = []
    with zipfile.ZipFile(celeba_path,"r") as z:
        for img_number in images_numbers:
            img_bytes = z.open('res_{}.jpg'.format((str(img_number)).zfill(6)))
            img = img_to_array(Image.open(img_bytes))
            images_dataset.append(img)
    images_dataset = np.array(images_dataset)
    return images_dataset

def plotHistory(hist_a, hist_b, leg_a, leg_b, title, x_label, y_label, filename):
    """ plot losses functions """
    plt.plot(hist_a)
    plt.plot(hist_b)
    plt.title(title)
    plt.ylabel(x_label)
    plt.xlabel(y_label)
    plt.legend([leg_a, leg_b], loc='upper left')
    plt.savefig(filename)
    plt.clf()

def plotHistoryThree(hist_a, hist_b, hist_c, leg_a, leg_b, leg_c, title, x_label, y_label, filename):
    """ plot losses functions """
    plt.plot(hist_a)
    plt.plot(hist_b)
    plt.plot(hist_c)
    plt.title(title)
    plt.ylabel(x_label)
    plt.xlabel(y_label)
    plt.legend([leg_a, leg_b, leg_c], loc='upper left')
    plt.savefig(filename)
    plt.clf()

def plotHistoryAcc(hist_a, hist_b, leg_a, leg_b, title, x_label, y_label, filename):
    """ plot accuracy functions """
    plt.plot(hist_a)
    plt.plot(hist_b)
    plt.title(title)
    plt.ylim(0, 1.0)
    plt.ylabel(x_label)
    plt.xlabel(y_label)
    plt.legend([leg_a, leg_b], loc='upper left')
    plt.savefig(filename)
    plt.clf()

def plotHistoryOne(hist_a, leg_a, title, x_label, y_label, filename):
    """ plot loss function """
    plt.plot(hist_a)
    plt.title(title)
    plt.ylabel(x_label)
    plt.xlabel(y_label)
    plt.legend([leg_a], loc='upper left')
    plt.savefig(filename)
    plt.clf()

def plotRealReconstruction(vae, test_dataset, line, column, filename, vae_type=False):
    m, n = line, column
    x = test_dataset[:column,:]
    reconstructions = vae.predict(x)
    if not vae_type:
      reconstructions = 0.5 * (reconstructions + 1)
      reconstructions = np.clip(reconstructions, 0, 1)
    for i in range(m):
        plt.figure(figsize=(14, 4))
        for j in range(n):
            plt.subplot(2, n,j +1)
            plt.imshow(np.squeeze(x[j]), cmap='gray')
            plt.axis('off')
            plt.subplot(2, n,1 + j + n)
            plt.imshow(np.squeeze(reconstructions[j]), cmap='gray')
            plt.axis('off')
        plt.savefig(filename)
        plt.close()

def sample_images(generator_model, run_folder, noise, vae_type=False):
    r, c = 5, 5
    gen_imgs = generator_model.predict(noise)
    if not vae_type:
      gen_imgs = 0.5 * (gen_imgs + 1)
      gen_imgs = np.clip(gen_imgs, 0, 1)
    fig, axs = plt.subplots(r, c, figsize=(20,20))
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(np.squeeze(gen_imgs[cnt, :,:,:]), cmap = 'gray')
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig(run_folder)
    plt.close()