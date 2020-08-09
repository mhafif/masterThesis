import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from datetime import datetime

import numpy as np
import tensorflow as tf
import tensorflow.keras as ks
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input, Flatten, Dense, Lambda, Reshape, BatchNormalization, LeakyReLU, ReLU, Dropout
from tensorflow.keras.utils import plot_model

from Common.Utils import plotHistory
from Common.Utils import plotHistoryOne
from Common.Utils import sample_images
from Common.Utils import plotRealReconstruction

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

def sample_z(args):  
    mu, sigma   = args  
    batch       = K.shape(mu)[0]  
    dim         = K.int_shape(mu)[1]  
    eps         = K.random_normal(shape=(batch, dim))  
    return mu + K.exp(sigma / 2) * eps

def build_models(enc_flt, dec_flt, dis_flt, enc_ker, dec_ker, dis_ker, activation, batchnorm, drop, drop_rate):
    """ Encoder model """
    # Input layer
    encoder_inputs = Input(shape=(SIZE, SIZE, NUM_CHANNELS), name='encoder_input')
    cx      = Conv2D(filters=enc_flt[0], kernel_size=enc_ker[0], strides=2, padding='same')(encoder_inputs)
    cx      = batchnorm_activation_dropout(cx, batchNorm=batchnorm, activation="relu", dropout=drop, dropRate=drop_rate)
    cx      = Conv2D(filters=enc_flt[1], kernel_size=enc_ker[1], strides=2, padding='same')(cx)
    cx      = batchnorm_activation_dropout(cx, batchNorm=batchnorm, activation="relu", dropout=drop, dropRate=drop_rate)
    x       = Flatten()(cx)
    x       = Dense(20)(x)
    x       = batchnorm_activation_dropout(x, batchNorm=batchnorm, activation="relu", dropout=drop, dropRate=drop_rate)
    mu      = Dense(LATENT_DIM, name='latent_mu')(x)
    sigma   = Dense(LATENT_DIM, name='latent_sigma')(x)
    # Get Conv2D shape for Conv2DTranspose operation in the decoder
    conv_shape = K.int_shape(cx)
    # Use reparameterization trick to ensure correct gradient
    z = Lambda(sample_z, output_shape=(LATENT_DIM, ), name='z')([mu, sigma])
    # Instantiate encoder
    encoder = Model(encoder_inputs, [mu, sigma, z], name='encoder')
    
    """ Decoder/Generator model """
    # Input layer
    decoder_inputs = Input(shape=(LATENT_DIM, ), name='decoder_input')
    x   = Dense(conv_shape[1] * conv_shape[2] * conv_shape[3])(decoder_inputs)
    x   = batchnorm_activation_dropout(x, batchNorm=batchnorm, activation="relu", dropout=drop, dropRate=drop_rate)
    x   = Reshape((conv_shape[1], conv_shape[2], conv_shape[3]))(x)
    # deconvolution layers
    cx  = Conv2DTranspose(filters=dec_flt[0], kernel_size=dec_ker[0], strides=2, padding='same')(x)
    cx  = batchnorm_activation_dropout(cx, batchNorm=batchnorm, activation="relu", dropout=drop, dropRate=drop_rate)
    cx  = Conv2DTranspose(filters=dec_flt[1], kernel_size=dec_ker[1], strides=2, padding='same')(cx)
    cx  = batchnorm_activation_dropout(cx, batchNorm=batchnorm, activation="relu", dropout=drop, dropRate=drop_rate)
    # decoder output layer
    decoded   = Conv2DTranspose(filters=dec_flt[2], kernel_size=dec_ker[2], activation='sigmoid', padding='same', name='decoder_output')(cx)
    # Instantiate decoder
    decoder = Model(decoder_inputs, decoded, name='decoder')
    
    """ Discriminator model """
    # Input layer
    discriminator_input = Input(shape=(SIZE, SIZE, NUM_CHANNELS), name='input_discriminator')
    # Convolution layers
    cl_1 = Conv2D(filters=dis_flt[0], kernel_size=dis_ker[0], strides=2, padding='same')(discriminator_input)
    cl_1 = batchnorm_activation_dropout(cl_1, batchNorm=batchnorm, activation=activation, dropout=drop, dropRate=drop_rate)
    cl_2 = Conv2D(filters=dis_flt[1], kernel_size=dis_ker[1], strides=2, padding='same')(cl_1)
    cl_3 = batchnorm_activation_dropout(cl_2, batchNorm=batchnorm, activation=activation, dropout=drop, dropRate=drop_rate)
    cl_4 = Conv2D(filters=dis_flt[2], kernel_size=dis_ker[2], strides=1, padding='same')(cl_3)
    cl_4 = batchnorm_activation_dropout(cl_4, batchNorm=batchnorm, activation=activation, dropout=drop, dropRate=drop_rate)
    x = Flatten()(cl_4)
    # Output layer
    output_discriminator = Dense(1, activation='sigmoid')(x)
    discriminator = Model(discriminator_input, [output_discriminator, x], name="Dicriminator")
    
    """ Sub-model: VAE """
    vae_outputs = decoder(encoder(encoder_inputs)[2])
    vae         = Model(encoder_inputs, vae_outputs, name='vae')

    """ VAE-GAN """
    vaegan_outputs = discriminator(vae(encoder_inputs))
    vaegan = Model(encoder_inputs, vaegan_outputs, name='vaegan')
    
    encoder.summary()
    decoder.summary()
    vae.summary()
    vaegan.summary()

    return encoder, decoder, discriminator, vae, vaegan

def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(-.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi), axis=raxis)
    
def compute_loss(x, encoder, decoder, discriminator, label_smooth_r, label_smooth_f):
    z_p = tf.random.normal(shape=(tf.shape(x)[0], LATENT_DIM))
    z_mean, z_log_var, z = encoder(x)
    out = decoder(z)
    x_p = decoder(z_p)
    dis_x_p, _ = discriminator(x_p)
    dis_x, dis_x_feat = discriminator(x)
    dis_x_tilde , dis_x_tilde_feat = discriminator(out)
    # kl loss computation 
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, z_mean, z_log_var)
    kl_loss = tf.reduce_mean(logqz_x - logpz)
    # gaussian perceptual loss as reconstruction loss computation
    reconstruction_loss = -tf.reduce_mean(log_normal_pdf(dis_x_feat, dis_x_tilde_feat, tf.zeros_like(dis_x_tilde_feat)))
    # gan loss computation
    cross_entropy = ks.losses.BinaryCrossentropy(from_logits=True)
    g_fake = cross_entropy(tf.ones_like(dis_x_tilde), dis_x_tilde)
    g_fake_p = cross_entropy(tf.ones_like(dis_x_p), dis_x_p)
    d_real = cross_entropy(tf.fill(tf.shape(dis_x), label_smooth_r), dis_x)
    d_fake = cross_entropy(tf.fill(tf.shape(dis_x_tilde), label_smooth_f), dis_x_tilde)
    d_fake_p = cross_entropy(tf.fill(tf.shape(dis_x_p), label_smooth_f), dis_x_p)
    return kl_loss, reconstruction_loss, g_fake + g_fake_p, d_fake + d_fake_p + d_real

def forwardBackwardBatch(x, encoder, decoder, discriminator, optimizer, gamma, label_smooth_r, label_smooth_f):
    with tf.GradientTape(persistent=True) as tape:
        l_kl, l_rec, l_gen, l_dis = compute_loss(x, encoder, decoder, discriminator, label_smooth_r, label_smooth_f)
        l_enc = l_kl + l_rec
        l_dec = (gamma * l_rec) - l_dis
    enc_grads = tape.gradient(l_enc, encoder.trainable_variables)
    optimizer[0].apply_gradients(zip(enc_grads, encoder.trainable_variables))
    dec_grads = tape.gradient(l_dec, decoder.trainable_variables)
    optimizer[1].apply_gradients(zip(dec_grads, decoder.trainable_variables))
    dis_grads = tape.gradient(l_dis, discriminator.trainable_variables)
    optimizer[2].apply_gradients(zip(dis_grads, discriminator.trainable_variables))
    return l_kl.numpy(), l_rec.numpy(), l_gen.numpy(), l_dis.numpy()

def train(x, encoder, decoder, discriminator, optimizer, gamma, label_smooth_r, label_smooth_f):
    loss_kl, loss_rec, loss_gen, loss_dis = 0.0, 0.0, 0.0, 0.0
    n = 0
    for train_x in x:
        a, b, c, d = forwardBackwardBatch(train_x, encoder, decoder, discriminator, optimizer, gamma, label_smooth_r, label_smooth_f)
        loss_kl += a
        loss_rec += b
        loss_gen += c
        loss_dis += d
        n += 1
    return loss_kl/n, loss_rec/n, loss_gen/n, loss_dis/n

def run_training(train_dataset, encoder, decoder, discriminator, vaegan, optimizers, gamma, noise, run_folder, label_smooth_r, label_smooth_f):
  enc_loss_hist = np.array([])
  dec_loss_hist = np.array([])
  dis_loss_hist = np.array([])
  for epoch in range(NO_EPOCHS):
      l = train(train_dataset, encoder, decoder, discriminator, optimizers, gamma, label_smooth_r, label_smooth_f)
      if (epoch+1) % 1 == 0:
          print('epoch:', epoch+1, 'kl_loss:', l[0], 'rec_loss:', l[1], 'gen_loss:', l[2], 'dis_loss:', l[3])
          enc_loss_hist = np.append(enc_loss_hist, (l[0]+l[1]))
          dec_loss_hist = np.append(dec_loss_hist, (l[1]-l[3]))
          dis_loss_hist = np.append(dis_loss_hist, l[3])
          sample_images(decoder, os.path.join(run_folder,"epochSamples","sample-%d" % (epoch+1)), noise, True)
          vaegan.save_weights(os.path.join(run_folder, "weights", "weights-%d.h5" % (epoch+1)))
  return enc_loss_hist, dec_loss_hist, dis_loss_hist

def fitting_process(encoder, decoder, discriminator, vae, vaegan, input_train, epoch, latent_dim, run_folder, lr_enc, lr_dec, lr_dis, gamma, label_smooth_r, label_smooth_f):
    # Fixed random latent space samples to plot progression
    noise = np.random.normal(0, 1, (5 * 5, latent_dim))
    # Training the vae-gan model
    optimizer_enc = ks.optimizers.RMSprop(lr_enc)
    optimizer_dec = ks.optimizers.RMSprop(lr_dec)
    optimizer_dis = ks.optimizers.RMSprop(lr_dis)
    enc_loss_hist, dec_loss_hist, dis_loss_hist = run_training(train_dataset, encoder, decoder, discriminator, vaegan,
                                                    [optimizer_enc, optimizer_dec, optimizer_dis], gamma, noise, run_folder, label_smooth_r, label_smooth_f)
    # Plotting the loss function of the gan model
    filename = os.path.join(run_folder,"VAEGAN_encdec_losses.png")
    plotHistory(enc_loss_hist, dec_loss_hist, "Encoder", "Decoder", "VAE-GAN Encoder-Decoder losses on MNIST", "loss", "epochs", filename)
    filename = os.path.join(run_folder,"VAEGAN_disc_losses.png")
    plotHistoryOne(dis_loss_hist, "Discriminator", "VAE-GAN discriminator loss on MNIST", "loss", "epochs", filename)
    # Plotting the visualisation of the models
    filename = os.path.join(run_folder,"encoder_vis.png")
    plot_model(encoder, to_file=filename, show_shapes=True, rankdir='LR')
    filename = os.path.join(run_folder,"decoder_vis.png")
    plot_model(decoder, to_file=filename, show_shapes=True, rankdir='LR')
    filename = os.path.join(run_folder,"discriminator_vis.png")
    plot_model(discriminator, to_file=filename, show_shapes=True, rankdir='LR')
    filename = os.path.join(run_folder,"vae_vis.png")
    plot_model(vae, to_file=filename, show_shapes=True, rankdir='LR')
    filename = os.path.join(run_folder,"vaegan_vis.png")
    plot_model(vaegan, to_file=filename, show_shapes=True, rankdir='LR')

def sampling_process(encoder, decoder, vaegan, run_folder, latent_dim, nb_samples, w_filename, datatype):
    # Load weights
    vaegan.load_weights(w_filename)
    # plot generated images
    if datatype == "mnist":
      size = 37
    else:
      size = 63
    mydpi=500
    noise = np.random.normal(0, 1, (nb_samples, latent_dim))
    images = decoder.predict(noise)
    for i in range(len(images)):
        filename = os.path.join(run_folder, "..", "..", "Evaluation", "mnistVAEGAN", "sample{}.jpg".format(i))
        plt.figure(figsize=(size/mydpi, size/mydpi), dpi=mydpi)
        plt.imshow(np.squeeze(images[i]), cmap='gray')
        plt.axis('off')
        plt.savefig(filename, bbox_inches='tight', pad_inches = 0, dpi=mydpi)
        plt.clf()

def plotting_process(vae, decoder, vaegan, input_test, run_folder, latent_dim, w_filename):
    # Load weights
    vaegan.load_weights(w_filename)
    # random latent space samples to plot
    noise = np.random.normal(0, 1, (5 * 5, latent_dim))
    sample_images(decoder, os.path.join(run_folder, "sample-model"), noise, True)
    # real vs reconstruction plot
    plotRealReconstruction(vae, input_test, 1, 5, os.path.join(run_folder, "realReconstructedSamples"), True)

if __name__ == "__main__":
    # Data & model configuration
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    CURRENT_DIR_RUN = os.path.join(CURRENT_DIR,"run_mnist")
    BATCH_SIZE = 128
    SIZE = 28
    LATENT_DIM = 32
    NUM_CHANNELS = 1
    NO_EPOCHS = 100

    # Load MNIST dataset
    (train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()
    train_images = train_images[:22400,:]
    test_images = test_images[:5600,:]
    # Parse numbers as floats
    train_images = train_images.reshape((-1,SIZE,SIZE,NUM_CHANNELS)).astype('float32')
    test_images = test_images.reshape((-1,SIZE,SIZE,NUM_CHANNELS)).astype('float32')
    # Normalize data
    train_images /= 255
    test_images /= 255
    # Create tensors of batches
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(1000).batch(BATCH_SIZE)
    test_dataset = test_images

    # Convolution filters and kernels configuration
    enc_filters = [64, 128]
    enc_kernels = [3, 3]
    dec_filters = [128, 64, NUM_CHANNELS]
    dec_kernels = [3, 3, 3]
    dis_filters = [32, 64, 128]
    dis_kernels = [3, 3, 3]

    GAMMA = 1.0
    LR_ENC = 0.0001
    LR_DEC = 0.0002
    LR_DIS = 0.0001

    batchnorm = True
    drop = False
    drop_rate = 0.1

    REAL_LABEL_SMOOTHING = 1
    FAKE_LABEL_SMOOTHING = 0

    # Building and compiling the models
    encoder, decoder, discriminator, vae, vaegan = build_models(enc_filters, dec_filters, dis_filters, enc_kernels, dec_kernels, 
                                                                dis_kernels, "leaky", batchnorm, drop, drop_rate)

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
        fitting_process(encoder, decoder, discriminator, vae, vaegan, train_dataset, NO_EPOCHS, LATENT_DIM, 
                CURRENT_DIR_RUN, LR_ENC, LR_DEC, LR_DIS, GAMMA, REAL_LABEL_SMOOTHING, FAKE_LABEL_SMOOTHING)
        
    if TASK == "sampling":
        nb_realsamples = 200
        sampling_process(encoder, decoder, vaegan, CURRENT_DIR_RUN, LATENT_DIM, nb_realsamples, weights_filename, "mnist")

    if TASK == "plotting":
        plotting_process(vae, decoder, vaegan, test_dataset, CURRENT_DIR_RUN, LATENT_DIM, weights_filename)

    print(datetime.now().strftime("%H:%M:%S"))