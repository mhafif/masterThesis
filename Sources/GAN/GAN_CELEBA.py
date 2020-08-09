import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from datetime import datetime

import numpy as np
from keras.layers import Conv2D, Conv2DTranspose, Input, Flatten, Dense, Reshape, Activation, BatchNormalization
from keras.models import Model
from keras.losses import binary_crossentropy
from keras import backend as K
from keras.optimizers import Adam
from keras.utils import plot_model

from Common.Utils import batchnorm_activation_dropout
from Common.Utils import get_celeba_dataset
from Common.Utils import plotHistory
from Common.Utils import plotHistoryThree
from Common.Utils import sample_images
from Common.Utils import normalInit

import matplotlib.pyplot as plt

def buildDiscriminator(dis_flt, dis_kern, input_shape, activation, batchnorm, drop, drop_rate):
    """
    Builds the discriminator model for the celeba dataset 
    where input_shape is the shape of the input data. Note 
    that a "cl" variable refers to a convolution layer
    """
    # Input layer
    input_discriminator = Input(shape=input_shape, name='input_discriminator')
    # Convolution layers
    cl_1 = Conv2D(dis_flt[0], dis_kern[0], strides=2, padding='same', kernel_initializer=normalInit(0.02))(input_discriminator)
    cl_1 = batchnorm_activation_dropout(cl_1, batchNorm=batchnorm, activation=activation, dropout=drop, dropRate=drop_rate)
    cl_2 = Conv2D(dis_flt[1], dis_kern[1], strides=2, padding='same', kernel_initializer=normalInit(0.02))(cl_1)
    cl_2 = batchnorm_activation_dropout(cl_2, batchNorm=batchnorm, activation=activation, dropout=drop, dropRate=drop_rate)
    cl_3 = Conv2D(dis_flt[2], dis_kern[2], strides=1, padding='same', kernel_initializer=normalInit(0.02))(cl_2)
    cl_3 = batchnorm_activation_dropout(cl_3, batchNorm=batchnorm, activation=activation, dropout=drop, dropRate=drop_rate)
    cl_4 = Conv2D(dis_flt[3], dis_kern[3], strides=1, padding='same', kernel_initializer=normalInit(0.02))(cl_3)
    cl_4 = batchnorm_activation_dropout(cl_4, batchNorm=batchnorm, activation=activation, dropout=drop, dropRate=drop_rate)
    x = Flatten()(cl_4)
    # Output layer
    output_discriminator = Dense(1, activation='sigmoid', kernel_initializer=normalInit(0.02))(x)
    discriminator_model = Model(input_discriminator, output_discriminator, name="Dicriminator")
    discriminator_model.summary()
    return discriminator_model

def buildGenerator(gen_flt, gen_kern, z_dim, size, gen_start_size, activation, batchnorm, drop, drop_rate):
    """
    Builds the generator model for the celeba dataset. Note 
    that a "dcl" variable refers to a deconvolution layer
    """
    # Input layer
    input_generator = Input(shape=(z_dim,), name='input_generator')
    # Deconvolution layers
    x = Dense(np.prod((size//8, size//8, gen_start_size)), kernel_initializer=normalInit(0.02))(input_generator)
    x = BatchNormalization()(x)
    x = Reshape((size//8, size//8, gen_start_size))(x)
    dcl_1 = Conv2DTranspose(gen_flt[0], gen_kern[0], strides=2, padding='same', kernel_initializer=normalInit(0.02))(x)
    dcl_1 = batchnorm_activation_dropout(dcl_1, batchNorm=batchnorm, activation=activation, dropout=drop, dropRate=drop_rate)
    dcl_2 = Conv2DTranspose(gen_flt[1], gen_kern[1], strides=2, padding='same', kernel_initializer=normalInit(0.02))(dcl_1)
    dcl_2 = batchnorm_activation_dropout(dcl_2, batchNorm=batchnorm, activation=activation, dropout=drop, dropRate=drop_rate)
    dcl_3 = Conv2DTranspose(gen_flt[2], gen_kern[2], strides=2, padding='same', kernel_initializer=normalInit(0.02))(dcl_2)
    dcl_3 = batchnorm_activation_dropout(dcl_3, batchNorm=batchnorm, activation=activation, dropout=drop, dropRate=drop_rate)
    dcl_4 = Conv2DTranspose(gen_flt[3], gen_kern[3], strides=1, padding='same', kernel_initializer=normalInit(0.02))(dcl_3)
    # Output layer
    output_generator = Activation('tanh')(dcl_4)
    generator_model = Model(input_generator, output_generator, name="Generator")
    generator_model.summary()
    return generator_model

def buildGAN(discriminator_model, generator_model, z_dim):
    """
    Builds the GAN model for the celeba dataset
    """
    discriminator_model.trainable = False
    gan_input = Input(shape=(z_dim,), name='GAN_input')
    gan_output = discriminator_model(generator_model(gan_input))
    gan_model = Model(gan_input, gan_output)
    gan_model.summary()
    return gan_model

def train_discriminator(gen_model, dis_model, true_imgs, batch_size, z_dim, label_noise_pct, label_smooth_r, label_smooth_f):
    """ 
    Train the discriminator with a batch of real and a batch of fake samples
    """
    random_idx = np.random.choice(np.arange(batch_size), replace=False, size=int(batch_size * label_noise_pct))
    # Real and fake train sets label with smoothing on real labels
    valid = np.full(shape=(batch_size,1), fill_value=label_smooth_r)
    fake = np.full(shape=(batch_size,1), fill_value=label_smooth_f)
    # Introduction of noise in fake and real labels with a percentage
    valid[random_idx] = label_smooth_f
    fake[random_idx] = label_smooth_r
    # Generation of the fake samples
    noise = np.random.normal(0, 1, (batch_size, z_dim))
    gen_imgs = gen_model.predict(noise)
    # Discriminator training on both real and fake sets
    d_loss_real = dis_model.train_on_batch(true_imgs, valid)
    d_loss_fake = dis_model.train_on_batch(gen_imgs, fake)
    # Recording of losses for real and fake sets 
    d_loss = 0.5 * (d_loss_real + d_loss_fake)
    return [d_loss, d_loss_real, d_loss_fake]

def train_generator(gan_model, batch_size, z_dim):
    """ 
    Train the gan model with fake generated sample all labeled real
    """
    valid = np.full(shape=(batch_size,1), fill_value=1)
    noise = np.random.normal(0, 1, (batch_size, z_dim))
    return gan_model.train_on_batch(noise, valid)

def train(gan_model, gen_model, dis_model, x_train, batch_size, no_epochs, run_folder, noise, z_dim, label_noise_pct, label_smooth_r, label_smooth_f):
    """
    train the gan model
    """
    gen_loss_hist = np.array([])
    dis_loss_hist = np.array([])
    dis_loss_real_hist = np.array([])
    dis_loss_fake_hist = np.array([])
    nb_batch = x_train.shape[0]//batch_size
    idx_train = np.arange(len(x_train))
    for epoch in range(1, no_epochs+1):
        np.random.shuffle(idx_train)
        genera_losses = np.array([])
        discri_losses = np.array([])
        discri_loss_r = np.array([])
        discri_loss_f = np.array([])
        for i in range(nb_batch):
            batch_x_train = x_train[idx_train[i*batch_size:(i+1)*batch_size]]
            d = train_discriminator(gen_model, dis_model, batch_x_train, len(batch_x_train), z_dim, label_noise_pct, label_smooth_r, label_smooth_f)
            g = train_generator(gan_model, batch_size, z_dim)
            genera_losses = np.append(genera_losses, g)
            discri_losses = np.append(discri_losses, d[0])
            discri_loss_r = np.append(discri_loss_r, d[1])
            discri_loss_f = np.append(discri_loss_f, d[2])
        gen_loss_hist = np.append(gen_loss_hist, np.mean(genera_losses))
        dis_loss_hist = np.append(dis_loss_hist, np.mean(discri_losses))
        dis_loss_real_hist = np.append(dis_loss_real_hist, np.mean(discri_loss_r))
        dis_loss_fake_hist = np.append(dis_loss_fake_hist, np.mean(discri_loss_f))
        print ("epoch: %d, [Dis_loss: %.3f (Dis_loss_real: %.3f, Dis_loss_fake: %.3f), Gen_loss: %.3f] " 
                % (epoch, np.mean(discri_losses), np.mean(discri_loss_r), np.mean(discri_loss_f), np.mean(genera_losses)))
        sample_images(gen_model, os.path.join(run_folder,"epochSamples","sample-%d" % (epoch)), noise)
        gan_model.save_weights(os.path.join(run_folder, "weights","weights-%d.h5" % (epoch)))
        epoch += 1
    return [gen_loss_hist, dis_loss_hist, dis_loss_real_hist, dis_loss_fake_hist]

def fitting_process(gan, gen, dis, input_train, epoch, latent_dim, noise_pct, label_smooth_r, label_smooth_f, run_folder):
    # Data & model configuration
    BATCH_SIZE = 128

    # Fixed random latent space samples to plot progression
    noise = np.random.normal(0, 1, (5 * 5, latent_dim))

    # Training the gan model    
    hist_array = train(gan, gen, dis, input_train, BATCH_SIZE, 
                epoch, run_folder, noise, latent_dim, noise_pct, label_smooth_r, label_smooth_f)
    
    # Plotting the loss function of the gan model
    gen_loss_hist = hist_array[0]
    dis_loss_hist = hist_array[1]
    dis_loss_real_hist = hist_array[2]
    dis_loss_fake_hist = hist_array[3]

    filename = os.path.join(run_folder,"GAN_losses.png")
    plotHistory(gen_loss_hist, dis_loss_hist, "Generator", "Discriminator", "GAN losses on CELEBA", "loss", "epochs", filename)
    filename = os.path.join(run_folder,"GAN_losses_r_f.png")
    plotHistoryThree(gen_loss_hist, dis_loss_real_hist, dis_loss_fake_hist, "Generator", "Discriminator real", "Discriminator fake", 
                    "GAN losses (Real vs Fake) on CELEBA", "loss", "epochs", filename)

    # Plotting the visualisation of the models
    filename = os.path.join(run_folder,"generator_vis.png")
    plot_model(gen, to_file=filename, show_shapes=True, rankdir='LR')
    filename = os.path.join(run_folder,"discriminator_vis.png")
    plot_model(dis, to_file=filename, show_shapes=True, rankdir='LR')
    filename = os.path.join(run_folder,"gan_vis.png")
    plot_model(gan, to_file=filename, show_shapes=True, rankdir='LR')

def sampling_process(dis, gen, gan, run_folder, latent_dim, nb_samples, w_filename, datatype):
    # Load weights
    gan.load_weights(w_filename)
    # plot generated images
    if datatype == "mnist":
      size = 37
    else:
      size = 63
    mydpi=500
    noise = np.random.normal(0, 1, (nb_samples, latent_dim))
    images = gen.predict(noise)
    images = 0.5 * (images + 1)
    images = np.clip(images, 0, 1)
    for i in range(len(images)):
        filename = os.path.join(run_folder, "..", "..", "Evaluation", "celebaGAN", "sample{}.jpg".format(i))
        plt.figure(figsize=(size/mydpi, size/mydpi), dpi=mydpi)
        plt.imshow(np.squeeze(images[i]))
        plt.axis('off')
        plt.savefig(filename, bbox_inches='tight', pad_inches = 0, dpi=mydpi)
        plt.clf()

def plotting_process(dis, gen, gan, run_folder, latent_dim, w_filename):
    # Load weights
    gan.load_weights(w_filename)
    # random latent space samples to plot
    noise = np.random.normal(0, 1, (5 * 5, latent_dim))
    sample_images(gen, os.path.join(run_folder, "sample-model"), noise)

if __name__ == "__main__":
    # Data & model configuration
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    CURRENT_DIR_RUN = os.path.join(CURRENT_DIR,"run_celeba")
    CELEBA_PATH = os.path.join(os.sep,'content','celeba{}.zip'.format(48))

    SIZE = 48
    LATENT_DIM = 100
    NUM_CHANNELS = 3
    NO_EPOCHS = 100
    input_shape = (SIZE, SIZE, NUM_CHANNELS)

    # Load CELEBA dataset
    dataset_size = 51200       # Maximum size is the whole dataset : 202599 
    train_test_split = 44800  
    celeba_dataset = get_celeba_dataset(dataset_size, 51200, CELEBA_PATH)
    input_train = celeba_dataset[:train_test_split,:]
    input_train = input_train[:22400,:]
    # Reshape data
    input_train = input_train.reshape(input_train.shape[0], SIZE, SIZE, NUM_CHANNELS)
    # Parse numbers as floats
    input_train = input_train.astype('float32')
    # Normalize data for a tanh activation function
    input_train = (input_train - 127.5) / 127.5

    # Model option: Convolution filters and kernels configuration, Dropout and batchnorm layer
    gen_filters = [256, 128, 64, NUM_CHANNELS]
    gen_kernels = [3, 3, 3, 3]
    dis_filters = [32, 64, 128, 256]
    dis_kernels = [3, 3, 3, 3]

    batchnorm = True
    drop = False
    drop_rate = 0.1

    GEN_START_SIZE = 128
    LABEL_NOISE_PCT = 0
    REAL_LABEL_SMOOTHING = 0.9
    FAKE_LABEL_SMOOTHING = 0

    # Building and compiling the models
    generator_model = buildGenerator(gen_filters, gen_kernels, LATENT_DIM, SIZE, GEN_START_SIZE, "relu", batchnorm, drop, drop_rate)
    discriminator_model = buildDiscriminator(dis_filters, dis_kernels, input_shape, "leaky", batchnorm, drop, drop_rate)
    opt_disc = Adam(lr=0.0001, beta_1=0.5)
    discriminator_model.compile(optimizer = opt_disc, loss = 'binary_crossentropy')
    gan_model = buildGAN(discriminator_model, generator_model, LATENT_DIM)
    opt_gan = Adam(lr=0.0002, beta_1=0.5)
    gan_model.compile(optimizer = opt_gan, loss='binary_crossentropy')

    # training, sampling, plotting
    weights_filename = None

    if sys.argv[1] == "-train":
      TASK = "training"
    elif sys.argv[1] == "-sample":
      TASK = "sampling"
      # training, sampling, plotting
      weights_filename = os.path.join(CURRENT_DIR_RUN, "weights", sys.argv[2])
    elif sys.argv[1] == "-plot":
      TASK = "plotting"
      # training, sampling, plotting
      weights_filename = os.path.join(CURRENT_DIR_RUN, "weights", sys.argv[2])
    else:
      print("Wrong arguments, choose between '-train', '-sample <weightFile>' and '-plot <weightFile>'")

    if TASK == "training":
        fitting_process(gan_model, generator_model, discriminator_model, input_train, NO_EPOCHS, 
                        LATENT_DIM, LABEL_NOISE_PCT, REAL_LABEL_SMOOTHING, FAKE_LABEL_SMOOTHING, CURRENT_DIR_RUN)
        
    if TASK == "sampling":
        nb_realsamples = 200
        sampling_process(discriminator_model, generator_model, gan_model, CURRENT_DIR_RUN, LATENT_DIM, nb_realsamples, weights_filename, "celeba")

    if TASK == "plotting":
        plotting_process(discriminator_model, generator_model, gan_model, CURRENT_DIR_RUN, LATENT_DIM, weights_filename)

    print(datetime.now().strftime("%H:%M:%S"))