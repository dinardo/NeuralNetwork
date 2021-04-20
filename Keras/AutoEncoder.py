"""
Example of Auto-Encoder

Course on Auto-Encoder
- https://blog.paperspace.com/autoencoder-image-compression-keras/

Needed libraries
- see file DeepNN.py
"""

import numpy             as np
import pandas            as pd
import tensorflow        as tf
import matplotlib.pyplot as plt

from keras.layers   import Input, Dense, Layer, LeakyReLU
from keras.models   import Model
from keras.datasets import mnist


#######################################
# Fix random seed for reproducibility #
#######################################
np.random.seed(3)
tf.random.set_seed(3)
plt.rcdefaults()


#################################################
# Load MNIST data, map gray scale 0-256 to 0-1, #
# and reshape the data to make it a vector      #
#################################################
(x_train_orig, y_train_orig), (x_test_orig, _) = mnist.load_data()

print('Train shape:', x_train_orig.shape)
print('Test shape:', x_test_orig.shape)

img_rows, img_cols = x_train_orig.shape[1], x_train_orig.shape[2]
x_train = x_train_orig.reshape(-1, img_rows * img_cols).astype('float32') / 255.
x_test  = x_test_orig.reshape (-1, img_rows * img_cols).astype('float32') / 255.


###########################
# Specify hyperparameters #
###########################
inOutDimension        = img_rows * img_cols
intermediateDimension = 256
latentDimension       = 2
batchSize             = 128
epochs                = 5


###########
# Encoder #
###########
encoderInput       = Input     (shape=(inOutDimension),      name='EncoderInput')
encoderDenseLayer1 = Dense     (units=intermediateDimension, name='encoderDdense1')   (encoderInput)
encoderActivLayer1 = LeakyReLU (                             name='encoderLeakyReLu1')(encoderDenseLayer1)
encoderDenseLayer2 = Dense     (units=latentDimension,       name='encoderDense2')    (encoderActivLayer1)
encoderOutput      = LeakyReLU (                             name='encoder_output')   (encoderDenseLayer2)

encoder = Model(encoderInput, encoderOutput, name='encoderModel')
encoder.summary()


###########
# Decoder #
###########
decoderInput       = Input     (shape=(latentDimension),     name='decoderInput')
decoderDenseLayer1 = Dense     (units=intermediateDimension, name='decoderDense1')    (decoderInput)
decoderActivLayer1 = LeakyReLU (                             name='decoderLeakyReLu1')(decoderDenseLayer1)
decoderDenseLayer2 = Dense     (units=inOutDimension,        name='decoderDense2')    (decoderActivLayer1)
decoderOutput      = LeakyReLU (                             name='decoderOutput')    (decoderDenseLayer2)

decoder = Model(decoderInput, decoderOutput, name='decoderModel')
decoder.summary()


################
# Auto-Encoder #
################
AEinput  = Input(shape=(inOutDimension), name='AEinput')
AElatent = encoder(AEinput)
AEoutput = decoder(AElatent)

AE = Model(AEinput, AEoutput, name='AE')
AE.summary()


##########################
# Compiling Auto-Encoder #
##########################
AE.compile(optimizer='adam', loss='mse')
tf.keras.utils.plot_model(AE, to_file='AE.png', show_shapes=True)


#########################
# Training Auto-Encoder #
#########################
history = AE.fit(x_train, x_train, epochs=epochs, batch_size=batchSize, shuffle=True, validation_data=(x_test, x_test))


###################
# Display history #
###################
def plotHistory(history):
    fig, ax = plt.subplots(figsize=(7,5))
    hist_df = pd.DataFrame(history.history)
    hist_df.plot(ax=ax)
    ax.set_ylabel('NELBO')
    ax.set_xlabel('# epochs')
    ax.set_ylim(.99 * hist_df[1:].values.min(),1.1 * hist_df[1:].values.max())
    plt.show()


#############################################
# Compare original and reconstructed images #
#############################################
def plotComparisonOriginal(encoder, decoder, data, images2show=5):
    img_rows, img_cols = data.shape[1], data.shape[2]

    encodedImages = encoder.predict(data.reshape(-1, img_rows * img_cols).astype('float32') / 255.)
    decodedImages = decoder(encodedImages)
    decodedImages = np.reshape(decodedImages, newshape=(-1, img_rows, img_cols))

    for indx in range(images2show):
        plotIndx = indx * 2 + 1

        plt.subplot(images2show, 2, plotIndx)
        plt.imshow(data[indx, :, :], cmap='Greys_r')

        plt.subplot(images2show, 2, plotIndx + 1)
        plt.imshow(decodedImages[indx, :, :], cmap='Greys_r')

    plt.show()


#################################################################
# Display how the latent space clusters different digit classes #
#################################################################
def plotLatentSpace(encoder, data, labels):
    img_rows, img_cols = data.shape[1], data.shape[2]

    encodedImages = encoder.predict(data.reshape(-1, img_rows * img_cols).astype('float32') / 255.)

    plt.figure(figsize=(7,5))
    plt.scatter(encodedImages[:, 0], encodedImages[:, 1], c=labels)
    plt.colorbar()
    plt.xlabel('z[0]')
    plt.ylabel('z[1]')
    plt.show()


############
# Plotting #
############
plotHistory(history)
plotComparisonOriginal(encoder, decoder, x_train_orig)
plotLatentSpace(encoder, x_train_orig, y_train_orig)


print('\n=== DONE ===')
