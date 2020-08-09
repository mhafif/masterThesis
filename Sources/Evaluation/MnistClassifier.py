import os
import keras
from keras.datasets import mnist
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, Conv2DTranspose, Input, Flatten, Dense, Reshape, Activation, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from numpy import argmax
import matplotlib.pyplot as plt

def preprocess_mnist(data):
    data = data.reshape(data.shape[0], 28, 28, 1)
    data = data.astype('float32')
    data /= 255
    return data

def plotHistory(hist_a, hist_b, leg_a, leg_b, title, x_label, y_label, filename, loca):
    """ plot functions """
    plt.plot(hist_a)
    plt.plot(hist_b)
    plt.title(title)
    if "Accuracy" in title:
      plt.ylim(0, 1.0)
    plt.ylabel(x_label)
    plt.xlabel(y_label)
    plt.legend([leg_a, leg_b], loc=loca)
    plt.savefig(filename)
    plt.clf()

def buildModel(input_shape=(28, 28, 1), num_classes=10):
    input_classifier = Input(shape=input_shape, name='inputClassifier')
    cl_1 = Conv2D(32, kernel_size=(3, 3), activation='relu')(input_classifier)
    cl_2 = Conv2D(64, kernel_size=(3, 3), activation='relu')(cl_1)
    cl_3 = MaxPooling2D(pool_size=(2, 2))(cl_2)
    cl_4 = Dropout(0.25)(cl_3)
    cl_5 = Flatten()(cl_4)
    cl_6 = Dense(128, activation='relu')(cl_5)
    cl_7 = Dropout(0.25)(cl_6)
    output = Dense(num_classes, activation='softmax')(cl_7)
    classifier_model = Model(input_classifier, output, name="Classifier")
    classifier_model_activation = Model(input_classifier, cl_6, name="Classifier")
    classifier_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
    return [classifier_model, classifier_model_activation]

def convert_to_class(predictions):
    return argmax(predictions, axis=-1)

if __name__ == "__main__":
    # current folder
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

    # training configuration
    batch_size = 128
    epochs = 20

    # input image dimensions
    SIZE = 28

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = preprocess_mnist(x_train)
    x_test = preprocess_mnist(x_test)

    # convert class vectors to binary class matrices
    num_classes = 10
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # build and train the model
    model, model_activation = buildModel()
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))
    # save weights
    model.save_weights(os.path.join(CURRENT_DIR, "mnistWeightsClassifier.h5"))

    # Evaluate model and print info
    loss_v, accuracy_v = model.evaluate(x_test, y_test, verbose=0)
    
    # Print values
    print('Test loss:', loss_v)
    print('Test accuracy:', accuracy_v)

    # Plots
    filename = os.path.join(CURRENT_DIR,"LossClassifier.png")
    plotHistory(history.history["loss"], history.history["val_loss"], "Train set", "Validation set", "Loss history on train and test set", "Epochs", "Loss", filename, 'upper right')
    filename = os.path.join(CURRENT_DIR,"AccuClassifier.png")
    plotHistory(history.history["accuracy"], history.history["val_accuracy"], "Train set", "Validation set", "Accuracy history on train and test set", "Epochs", "Accuracy", filename, 'lower right')











