import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from datetime import datetime

import numpy as np
from keras.layers import Conv2D, Conv2DTranspose, Input, Flatten, Dense, Lambda, Reshape
from keras.models import Model
from keras.datasets import mnist
from keras.losses import binary_crossentropy
from keras import backend as K
from keras.utils import plot_model

from Common.Utils import batchnorm_activation_dropout
from Common.Utils import plotHistory
from Common.Utils import sample_images
from Common.Utils import plotRealReconstruction
from Common.Utils import plotHistoryAcc

import matplotlib.pyplot as plt

def sample_z(args):
    """ Sampling with reparameterization trick """
    mu, sigma   = args  
    batch       = K.shape(mu)[0]  
    dim         = K.int_shape(mu)[1]  
    eps         = K.random_normal(shape=(batch, dim))  
    return mu + K.exp(sigma / 2) * eps

def buildModels(enc_flt, dec_flt, enc_ker, dec_ker, z_dim, input_shape, activation, batchnorm, drop, drop_rate):
    """ ENCODER MODEL """
    # encoder input layer
    encoder_inputs = Input(shape=input_shape, name='encoder_input') 
    # convolution layers
    cx      = Conv2D(filters=enc_flt[0], kernel_size=enc_ker[0], strides=2, padding='same')(encoder_inputs)
    cx      = batchnorm_activation_dropout(cx, batchNorm=batchnorm, activation=activation, dropout=drop, dropRate=drop_rate)
    cx      = Conv2D(filters=enc_flt[1], kernel_size=enc_ker[1], strides=2, padding='same')(cx)
    cx      = batchnorm_activation_dropout(cx, batchNorm=batchnorm, activation=activation, dropout=drop, dropRate=drop_rate)
    x       = Flatten()(cx)
    x       = Dense(20)(x)
    x       = batchnorm_activation_dropout(x, batchNorm=batchnorm, activation=activation, dropout=drop, dropRate=drop_rate)
    # latent space mu and sigma vectors
    mu      = Dense(z_dim, name='latent_mu')(x)
    sigma   = Dense(z_dim, name='latent_sigma')(x)
    # save Conv2D shape for Conv2DTranspose in decoder
    conv_shape = K.int_shape(cx)
    # Use reparameterization trick to ensure correct gradient
    z = Lambda(sample_z, output_shape=(z_dim, ), name='z')([mu, sigma])
    # Instantiate encoder
    encoder = Model(encoder_inputs, [mu, sigma, z], name='Encoder')

    """ DECODER MODEL """
    # decoder input layer
    decoder_inputs = Input(shape=(z_dim, ), name='decoder_input')
    x   = Dense(conv_shape[1] * conv_shape[2] * conv_shape[3])(decoder_inputs)
    x   = batchnorm_activation_dropout(x, batchNorm=batchnorm, activation=activation, dropout=drop, dropRate=drop_rate)
    x   = Reshape((conv_shape[1], conv_shape[2], conv_shape[3]))(x)
    # deconvolution layers
    cx  = Conv2DTranspose(filters=dec_flt[0], kernel_size=dec_ker[0], strides=2, padding='same')(x)
    cx  = batchnorm_activation_dropout(cx, batchNorm=batchnorm, activation=activation, dropout=drop, dropRate=drop_rate)
    cx  = Conv2DTranspose(filters=dec_flt[1], kernel_size=dec_ker[1], strides=2, padding='same')(cx)
    cx  = batchnorm_activation_dropout(cx, batchNorm=batchnorm, activation=activation, dropout=drop, dropRate=drop_rate)
    # decoder output layer
    decoder_output   = Conv2DTranspose(filters=dec_flt[2], kernel_size=dec_ker[2], activation='sigmoid', padding='same', name='decoder_output')(cx)
    # Instantiate decoder
    decoder = Model(decoder_inputs, decoder_output, name='Decoder')

    """ Kullback leibler divergence and reconstruction loss """
    def kl_reconstruction_loss(true, pred):  
        # Reconstruction loss  
        reconstruction_loss = binary_crossentropy(K.flatten(true), K.flatten(pred)) * SIZE * SIZE 
        # KL divergence loss  
        kl_loss = 1 + sigma - K.square(mu) - K.exp(sigma) 
        kl_loss = K.sum(kl_loss, axis=-1)  
        kl_loss *= -0.5  
        # Total loss = 50% rec + 50% KL divergence loss
        return K.mean(reconstruction_loss + kl_loss)

    """ VAE MODEL """
    vae_output  = decoder(encoder(encoder_inputs)[2])
    vae         = Model(encoder_inputs, vae_output, name='VAE')

    """ Compile the model """
    vae.compile(optimizer='adam', loss=kl_reconstruction_loss, metrics=['accuracy'])
    
    """ Models summary print """
    encoder.summary()
    decoder.summary()
    vae.summary()

    return encoder, decoder, vae

def train(vae_model, decoder, train_set, validation_set, batch_size, no_epochs, run_folder, noise):
    """ Train the model and return loss and accuracy of train and validation set """
    train_loss_hist = np.array([])
    train_acc_hist = np.array([])
    validation_loss_hist = np.array([])
    validation_acc_hist = np.array([])
    nb_batch = train_set.shape[0]//batch_size
    idx_train = np.arange(len(train_set))
    for epoch in range(1, no_epochs+1):
        np.random.shuffle(idx_train)
        batch_losses = np.array([])
        batch_acc = np.array([])
        for i in range(nb_batch):
            batch_x_train = train_set[idx_train[i*batch_size:(i+1)*batch_size]]
            vae_loss, vae_acc = vae_model.train_on_batch(batch_x_train, batch_x_train)
            batch_losses = np.append(batch_losses, vae_loss)
            batch_acc = np.append(batch_acc, vae_acc)
        train_epoch_loss, train_epoch_acc = np.mean(batch_losses), np.mean(batch_acc)
        validation_loss, validation_acc = vae_model.evaluate(validation_set, validation_set, verbose=0)
        train_loss_hist = np.append(train_loss_hist, train_epoch_loss)
        train_acc_hist = np.append(train_acc_hist, train_epoch_acc)
        validation_loss_hist = np.append(validation_loss_hist, validation_loss)
        validation_acc_hist = np.append(validation_acc_hist, validation_acc)
        
        print ("epoch: %d, [TrainSet loss: %.3f, TrainSet accuracy: %.3f] [ValidSet loss: %.3f, ValidSet accuracy: %.3f] " 
          % (epoch, train_epoch_loss, train_epoch_acc, validation_loss, validation_acc))
        sample_images(decoder, os.path.join(run_folder,"epochSamples","sample-%d" % (epoch)), noise, True)
        vae_model.save_weights(os.path.join(run_folder, "weights", "weights-%d.h5" % (epoch)))
        epoch += 1
    return train_loss_hist, train_acc_hist, validation_loss_hist, validation_acc_hist

def fitting_process(encoder, decoder, vae, input_train, input_test, epoch, latent_dim, run_folder):
    # Data & model configuration
    BATCH_SIZE = 128

    # Fixed random latent space samples to plot progression
    noise = np.random.normal(0, 1, (5 * 5, latent_dim))
    # Training the vae model
    hist_array = train(vae, decoder, input_train, input_test, BATCH_SIZE, epoch, run_folder, noise)

    # Plotting the loss function and the accuracy of the vae model
    train_loss_hist = hist_array[0]
    train_acc_hist = hist_array[1]
    validation_loss_hist = hist_array[2]
    validation_acc_hist = hist_array[3]
    filename = os.path.join(run_folder,"vae_loss.png")
    plotHistory(train_loss_hist, validation_loss_hist, "train set", "validation set", "VAE loss on MNIST", "loss", "epochs", filename)
    filename = os.path.join(run_folder,"vae_accuracy.png")
    plotHistoryAcc(train_acc_hist, validation_acc_hist, "train set", "validation set", "VAE accuracy on MNIST", "accuracy", "epochs", filename)

    # Plotting the visualisation of the models
    filename = os.path.join(run_folder,"encoder_vis.png")
    plot_model(encoder, to_file=filename, show_shapes=True, rankdir='LR')
    filename = os.path.join(run_folder,"decoder_vis.png")
    plot_model(decoder, to_file=filename, show_shapes=True, rankdir='LR')
    filename = os.path.join(run_folder,"vae_vis.png")
    plot_model(vae, to_file=filename, show_shapes=True, rankdir='LR')

def sampling_process(encoder, decoder, vae, run_folder, latent_dim, nb_samples, w_filename, datatype):
    # Load weights
    vae.load_weights(w_filename)
    # plot generated images
    if datatype == "mnist":
      size = 37
    else:
      size = 63
    mydpi=500
    noise = np.random.normal(0, 1, (nb_samples, latent_dim))
    images = decoder.predict(noise)
    for i in range(len(images)):
        filename = os.path.join(run_folder, "..", "..", "Evaluation", "mnistVAE", "sample{}.jpg".format(i))
        plt.figure(figsize=(size/mydpi, size/mydpi), dpi=mydpi)
        plt.imshow(np.squeeze(images[i]), cmap='gray')
        plt.axis('off')
        plt.savefig(filename, bbox_inches='tight', pad_inches = 0, dpi=mydpi)
        plt.clf()

def plotting_process(encoder, decoder, vae, input_test, run_folder, latent_dim, w_filename):
    # Load weights
    vae.load_weights(w_filename)
    # random latent space samples to plot
    noise = np.random.normal(0, 1, (5 * 5, latent_dim))
    sample_images(decoder, os.path.join(run_folder, "sample-model"), noise, True)
    # real vs reconstruction plot
    plotRealReconstruction(vae, input_test, 1, 5, os.path.join(run_folder, "realReconstructedSamples"), True)

if __name__ == "__main__":
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    CURRENT_DIR_RUN = os.path.join(CURRENT_DIR,"run_mnist")
    SIZE = 28
    LATENT_DIM = 32
    NUM_CHANNELS = 1
    NO_EPOCHS = 100
    input_shape = (SIZE, SIZE, NUM_CHANNELS)

    # Load MNIST dataset
    (input_train, _), (input_test, _) = mnist.load_data()
    input_train = input_train[:22400,:]
    input_test = input_test[:5600,:]
    # Reshape data
    input_train = input_train.reshape(input_train.shape[0], SIZE, SIZE, NUM_CHANNELS)
    input_test = input_test.reshape(input_test.shape[0], SIZE, SIZE, NUM_CHANNELS)
    # Parse numbers as floats
    input_train = input_train.astype('float32')
    input_test = input_test.astype('float32')
    # Normalize data
    input_train = input_train / 255
    input_test = input_test / 255

    # Model option: Convolution filters and kernels configuration, Dropout and batchnorm layer
    enc_filters = [64, 128]
    enc_kernels = [3, 3]
    dec_filters = [128, 64, NUM_CHANNELS]
    dec_kernels = [3, 3, 3]

    batchnorm = True
    drop = False
    drop_rate = 0.1

    # Building and compiling the models
    encoder, decoder, vae = buildModels(enc_filters, dec_filters, enc_kernels, dec_kernels, LATENT_DIM, input_shape, "relu", batchnorm, drop, drop_rate)

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
        fitting_process(encoder, decoder, vae, input_train, input_test, NO_EPOCHS, LATENT_DIM, CURRENT_DIR_RUN)
        
    if TASK == "sampling":
        nb_realsamples = 200
        sampling_process(encoder, decoder, vae, CURRENT_DIR_RUN, LATENT_DIM, nb_realsamples, weights_filename, "mnist")

    if TASK == "plotting":
        plotting_process(encoder, decoder, vae, input_test, CURRENT_DIR_RUN, LATENT_DIM, weights_filename)

    print(datetime.now().strftime("%H:%M:%S"))