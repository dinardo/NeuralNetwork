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

from tensorflow.keras.layers   import Input, Dense, Layer, LeakyReLU
from tensorflow.keras.models   import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.backend  import get_value


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
epochs                = 30


###########
# Encoder #
###########
encoderInput       = Input     (shape=(inOutDimension),      name='encoderInput')

encoderDenseLayer  = Dense     (units=intermediateDimension, name='encoderDense1')     (encoderInput)
encoderActivLayer  = LeakyReLU (                             name='encoderLeakyReLu1') (encoderDenseLayer)
encoderDenseLayer  = Dense     (units=intermediateDimension, name='encoderDense2')     (encoderActivLayer)
encoderActivLayer  = LeakyReLU (                             name='encoderLeakyReLu2') (encoderDenseLayer)
encoderDenseLayer  = Dense     (units=intermediateDimension, name='encoderDense3')     (encoderActivLayer)
encoderActivLayer  = LeakyReLU (                             name='encoderLeakyReLu3') (encoderDenseLayer)

encoderLatentLayer = Dense     (units=latentDimension,       name='encoderLatentLayer')(encoderActivLayer)
encoderOutput      = LeakyReLU (                             name='encoderOutput')     (encoderLatentLayer)

encoder = Model(encoderInput, encoderOutput, name='encoderModel')
encoder.summary()


###########
# Decoder #
###########
decoderInput       = Input     (shape=(latentDimension),     name='decoderInput')

decoderDenseLayer  = Dense     (units=intermediateDimension, name='decoderDense1')     (decoderInput)
decoderActivLayer  = LeakyReLU (                             name='decoderLeakyReLu1') (decoderDenseLayer)
decoderDenseLayer  = Dense     (units=intermediateDimension, name='decoderDense2')     (decoderActivLayer)
decoderActivLayer  = LeakyReLU (                             name='decoderLeakyReLu2') (decoderDenseLayer)
decoderDenseLayer  = Dense     (units=intermediateDimension, name='decoderDense3')     (decoderActivLayer)
decoderActivLayer  = LeakyReLU (                             name='decoderLeakyReLu3') (decoderDenseLayer)

decoderLatentLayer = Dense     (units=inOutDimension,        name='decoderLatentLayer')(decoderActivLayer)
decoderOutput      = LeakyReLU (                             name='decoderOutput')     (decoderLatentLayer)

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
from tensorflow.keras.utils import plot_model
AE.compile(optimizer='adam', loss='mse')
plot_model(AE, to_file='AE.png', show_shapes=True, expand_nested=True)


#######
# GPU #
#######
print('--> Number of available GPUs:', len(tf.config.list_physical_devices('GPU')))

if tf.config.list_physical_devices('GPU'):
  strategy = tf.distribute.MirroredStrategy()
else: # Use the Default Strategy
  strategy = tf.distribute.get_strategy()

print('--> Stragegy:', strategy)

with strategy.scope():
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


############################
# Display a grid of digits #
############################
def plotGeneration(decoder, img_rows, img_cols, scale=14.0, n=15):
    figure = np.zeros((img_rows * n, img_cols * n))
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            sampleLatentSpace = np.array([[xi, yi]])
            decodedImage      = decoder.predict(sampleLatentSpace)
            decodedImage      = np.reshape(decodedImage, newshape=(-1, img_rows, img_cols))
            figure[i * img_rows : (i + 1) * img_rows, j * img_cols : (j + 1) * img_cols] = decodedImage

    plt.figure(figsize=(7,5))
    start_range    = img_rows // 2
    end_range      = n * img_rows + start_range
    pixel_range    = np.arange(start_range, end_range, img_rows)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)

    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel('z[0]')
    plt.ylabel('z[1]')
    plt.imshow(figure, cmap='Greys_r')
    plt.show()


#############################################
# Compare original and reconstructed images #
#############################################
def plotComparisonOriginal(decodedImages, data, images2show=5):
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
def plotLatentSpace(encodedImages, labels):
    plt.figure(figsize=(7,5))
    plt.scatter(encodedImages[:, 0], encodedImages[:, 1], c=labels)
    plt.colorbar()
    plt.xlabel('z[0]')
    plt.ylabel('z[1]')
    plt.show()


############
# Plotting #
############
from tensorflow.keras.losses import mse
encodedImages = encoder.predict(x_train)
decodedImages = decoder(encodedImages)
print('Image distance between original and reconstructed:', get_value(mse(x_train[0], decodedImages[0])))
decodedImages = np.reshape(decodedImages, newshape=(-1, img_rows, img_cols))

plotHistory(history)
plotGeneration(decoder, img_rows, img_cols)
plotComparisonOriginal(decodedImages, x_train_orig)
plotLatentSpace(encodedImages, y_train_orig)


print('\n=== DONE ===')
