import keras 
from keras.layers import GaussianNoise, Conv2D, Conv2DTranspose, Input, Flatten, Dense, Lambda, Reshape, Dropout, Activation
from keras.layers import BatchNormalization, LeakyReLU, UpSampling2D, ReLU
from keras.models import Model
from keras.datasets import mnist
from keras.losses import binary_crossentropy
from keras import backend as K
from keras.optimizers import Adam
from keras.preprocessing.image import array_to_img, img_to_array, load_img
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import glob

def get_celeba_dataset(size_training, dataset_size):
    if dataset_size > 202599:
        raise ValueError('Celeba dataset has only 202599 images')
    if size_training > dataset_size:
        raise ValueError('Not enough images available for a training set of size {}'.format(size_training)) 
    images_numbers = np.arange(1,dataset_size+1)
    np.random.shuffle(images_numbers)
    images_numbers = images_numbers[:size_training]
    images_dataset = []
    for img_number in images_numbers:
        img_number_str = (str(img_number)).zfill(6)
        img = img_to_array(load_img(os.path.join(CELEBA_PATH,'res_{}.jpg'.format(img_number_str))))
        images_dataset.append(img)
    images_dataset = np.array(images_dataset)
    return images_dataset

def normalInit(stdv):
    return keras.initializers.RandomNormal(stddev=stdv)

IMG_DIM = 48
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CURRENT_DIR_RUN = os.path.join(CURRENT_DIR,"run")
CELEBA_PATH = os.path.join(os.sep,'content','celeba{}'.format(IMG_DIM))

# Data & model configuration
dataset_size = 51200 # Maximum size 202599
verbosity = 1
num_channels = 3

batch_size = 128
no_epochs = 100
dim_z = 100

filter_size_gen = 128
filter_size_dis = 128
kernel_size_gen = 3
kernel_size_dis = 3
gen_start_size = 128
label_noise_pct = 0
valid_label_smoothing = 1
gen_overtrain_factor = 1
dis_penality = np.concatenate((np.arange(1,0.5,-0.5/(no_epochs/2)),np.arange(0.5,1,0.5/(no_epochs/2))))  #0.75
noise = np.random.normal(0, 1, (5 * 5, dim_z)) #from which image is generated for each epoch

# image set and image preprocesing
input_train = get_celeba_dataset(dataset_size,51200) # Maximum size 202599
img_width, img_height = input_train.shape[1], input_train.shape[2]
input_shape = (img_height, img_width, num_channels)
input_train = input_train.astype('float32')
input_train = (input_train - 127.5) / 127.5

# Discriminator Definition
input_discriminator = Input(shape=input_shape, name='input_discriminator')

cx1 = Conv2D(filters=filter_size_dis, kernel_size=kernel_size_dis, strides=2, padding='same', kernel_initializer=normalInit(0.02))(input_discriminator)
#cx1 = GaussianNoise(stddev=0.02)(cx1)
cx2 = Conv2D(filters=filter_size_dis, kernel_size=kernel_size_dis, strides=2, padding='same', kernel_initializer=normalInit(0.02))(cx1)
cx2 = LeakyReLU(alpha=0.2)(cx2)
cx3 = Conv2D(filters=filter_size_dis, kernel_size=kernel_size_dis, strides=2, padding='same', kernel_initializer=normalInit(0.02))(cx2)
cx3 = BatchNormalization()(cx3)
cx3 = LeakyReLU(alpha=0.2)(cx3)
cx4 = Conv2D(filters=filter_size_dis, kernel_size=kernel_size_dis, strides=1, padding='same', kernel_initializer=normalInit(0.02))(cx3)
cx4 = BatchNormalization()(cx4)
cx4 = LeakyReLU(alpha=0.2)(cx4)
x = Flatten()(cx4)

output_discriminator = Dense(1, activation='sigmoid', kernel_initializer=normalInit(0.02))(x)
discriminator_model = Model(input_discriminator, output_discriminator, name="Dicriminator")
discriminator_model.summary()

# Generator Definition
input_generator = Input(shape=(dim_z,), name='input_generator')

x = Dense(np.prod((img_height//8,img_width//8,gen_start_size)), kernel_initializer=normalInit(0.02))(input_generator)
x = BatchNormalization()(x)
x = Reshape((img_height//8,img_width//8,gen_start_size))(x)
cx1 = Conv2DTranspose(filters=filter_size_gen, kernel_size=kernel_size_gen, strides=2, padding='same', kernel_initializer=normalInit(0.02))(x)
cx1 = BatchNormalization()(cx1)
cx1 = LeakyReLU(alpha=0.2)(cx1)
cx2 = Conv2DTranspose(filters=filter_size_gen, kernel_size=kernel_size_gen, strides=2, padding='same', kernel_initializer=normalInit(0.02))(cx1)
cx2 = BatchNormalization()(cx2)
cx2 = LeakyReLU(alpha=0.2)(cx2)
cx3 = Conv2DTranspose(filters=filter_size_gen, kernel_size=kernel_size_gen, strides=2, padding='same', kernel_initializer=normalInit(0.02))(cx2)
cx3 = BatchNormalization()(cx3)
cx3 = LeakyReLU(alpha=0.2)(cx3)
cx4 = Conv2DTranspose(filters=num_channels, kernel_size=kernel_size_gen, strides=1, padding='same', kernel_initializer=normalInit(0.02))(cx3)

output_generator = Activation('tanh')(cx4)
generator_model = Model(input_generator, output_generator, name="Generator")
generator_model.summary()

# Optimizers
# - Dis: 0.0002 | Gen: 0.0002 faster learning from discriminator
# - Dis: 0.0001 | Gen: 0.0002 even faster learning from discriminator
# - Dis: 0.0002 | Gen: 0.0001 slow learning from the discriminator
opt_disc = Adam(lr=0.0001, beta_1=0.5)
opt_gan = Adam(lr=0.0002, beta_1=0.5)

# Discriminator compilation
discriminator_model.compile(optimizer = opt_disc, loss = 'binary_crossentropy', metrics = ['accuracy'])
        
# GAN compilation
discriminator_model.trainable = False
gan_input = Input(shape=(dim_z,), name='GAN_input')
gan_output = discriminator_model(generator_model(gan_input))
gan_model = Model(gan_input, gan_output)
gan_model.summary()
gan_model.compile(optimizer = opt_gan, loss='binary_crossentropy', metrics=['accuracy'])

def train_discriminator(generator_model, discriminator_model, true_imgs, batch_size, only_real):
    random_idx = np.random.choice(np.arange(batch_size), replace=False, size=int(batch_size * label_noise_pct))
    if only_real:
      valid = np.full(shape=(batch_size,1),fill_value=valid_label_smoothing)
      valid[random_idx] = 0
      d_loss_real, d_acc_real = discriminator_model.train_on_batch(true_imgs, valid)
    else:
      # Real and fake train sets label with smoothing on labels
      valid = np.full(shape=(batch_size,1),fill_value=valid_label_smoothing)
      fake = np.full(shape=(batch_size,1),fill_value=0)
      # Noise in label introduction with a percentage
      valid[random_idx] = 0
      fake[random_idx] = valid_label_smoothing
      # Fake train set generation by the generator
      noise = np.random.normal(0, 1, (batch_size, dim_z))
      gen_imgs = generator_model.predict(noise)
      # Discriminator training on both real and fake sets
      d_loss_real, d_acc_real = discriminator_model.train_on_batch(true_imgs, valid)
      d_loss_fake, d_acc_fake = discriminator_model.train_on_batch(gen_imgs, fake)
      # Recording of losses for real and fake sets
      d_loss =  0.5 * (d_loss_real + d_loss_fake)
      d_acc = 0.5 * (d_acc_real + d_acc_fake)
      return [d_loss, d_loss_real, d_loss_fake, d_acc, d_acc_real, d_acc_fake]

def train_generator(gan_model, batch_size):
    overtrain_batch_size = int(batch_size * gen_overtrain_factor)
    valid = np.full(shape=(overtrain_batch_size,1),fill_value=1)
    noise = np.random.normal(0, 1, (overtrain_batch_size, dim_z))
    return gan_model.train_on_batch(noise, valid)

def train(gan_model, generator_model, x_train, batch_size, no_epochs, run_folder):
    g = [0, dis_penality[0]]
    d = [0,0,0,0.9,0,0]
    nb_batch = x_train.shape[0]//batch_size
    idx_train = np.arange(len(x_train))
    for epoch in range(1, no_epochs+1):
        np.random.shuffle(idx_train)
        discri_losses = np.array([])
        discri_acc = np.array([])
        discri_acc_r = np.array([])
        discri_acc_f = np.array([])
        genera_losses = np.array([])
        genera_acc = np.array([])
        dis_pct_train = 0
        for i in range(nb_batch):
            batch_x_train = x_train[idx_train[i*batch_size:(i+1)*batch_size]]
            if g[1] >= dis_penality[epoch-1]:
              dis_pct_train += 1
              d = train_discriminator(generator_model, discriminator_model, batch_x_train, len(batch_x_train), False)
              discri_losses = np.append(discri_losses, d[0])
              discri_acc = np.append(discri_acc, d[3])
              discri_acc_r = np.append(discri_acc_r, d[4])
              discri_acc_f = np.append(discri_acc_f, d[5])
            # else:
            #   train_discriminator(generator_model, discriminator_model, batch_x_train, len(batch_x_train), True)
            g = train_generator(gan_model, batch_size)
            genera_losses = np.append(genera_losses, g[0])
            genera_acc = np.append(genera_acc, g[1])
        print ("epoch: %d, penality: %.4f, disTrainingPercentage: %.1f, [D_loss: %.3f, D_acc: %.3f (acc_real: %.3f, acc_fake: %.3f)] [G_loss: %.3f, G_acc: %.3f] " 
          % (epoch, dis_penality[epoch-1], (dis_pct_train/nb_batch)*100, np.mean(discri_losses), np.mean(discri_acc), np.mean(discri_acc_r), np.mean(discri_acc_f), np.mean(genera_losses), np.mean(genera_acc)))
        sample_images(generator_model, os.path.join(CURRENT_DIR_RUN,"sample-%d" % (epoch)),noise)
        # gan_model.save_weights(os.path.join(CURRENT_DIR_RUN, "weights","weights-%d.h5" % (epoch)))
        # gan_model.save_weights(os.path.join(CURRENT_DIR_RUN, "weights","weights.h5"))
        epoch += 1
    
def load_weights(filepath):
    gan_model.load_weights(filepath)

def sample_images(generator_model, run_folder, noise):
    r, c = 5, 5
    gen_imgs = generator_model.predict(noise)
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

train(gan_model, generator_model, input_train,batch_size = batch_size, no_epochs = no_epochs, run_folder = CURRENT_DIR)