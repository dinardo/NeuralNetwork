"""
Example of Variational Auto-Encoder

Courses on Variational Auto-Encoder
- https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf
- https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73

Needed libraries
- see file DeepNN.py
"""

import numpy             as np
import pandas            as pd
import tensorflow        as tf
import matplotlib.pyplot as plt
import keras.backend     as k_be

from keras.layers   import BatchNormalization, Input, Conv2D, Conv2DTranspose, Flatten, Dense, Lambda, Layer, Reshape
from keras.models   import Model
from keras.datasets import mnist
from keras.backend  import get_value
from keras.losses   import binary_crossentropy


#######################################
# Fix random seed for reproducibility #
#######################################
np.random.seed(3)
tf.random.set_seed(3)
plt.rcdefaults()


###################
# Hyperparameters #
###################
latentDimension = 2
batchSize       = 128
epochs          = 30
nChns           = 1


#####################################################
# Load MNIST data, map gray scale 0-256 to 0-1, and #
# reshape the data to make it a vector              #
#####################################################
(x_train_orig, y_train_orig), (x_test_orig, _) = mnist.load_data()

print('Train shape:', x_train_orig.shape)
print('Test shape:', x_test_orig.shape)

img_rows, img_cols = x_train_orig.shape[1], x_train_orig.shape[2]
x_train = x_train_orig.reshape(-1, img_rows, img_cols, nChns).astype('float32') / 255
x_test  = x_test_orig.reshape(-1,  img_rows, img_cols, nChns).astype('float32') / 255


#########################
# Sampling latent space #
#########################
def sampleLatentSpace(inputs):
    mean, log_var = inputs
    batch         = tf.shape(mean)[0]
    dim           = tf.shape(mean)[1]
    epsilon       = k_be.random_normal(shape=(batch, dim))

    return mean + k_be.exp(log_var / 2) * epsilon


#######################
# Reconstruction loss #
#######################
def reconstructionLoss(true, pred):
    return k_be.sum(binary_crossentropy(k_be.batch_flatten(true), k_be.batch_flatten(pred)), axis=-1)


###############################
# Kullback-Leibler loss layer #
###############################
class KullbackLeiblerDivergenceLayer(Layer):
    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KullbackLeiblerDivergenceLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):
        mean, log_var = inputs
        klLoss = k_be.mean(-0.5 * k_be.sum(1 + log_var - k_be.square(mean) - k_be.exp(log_var), axis=-1))

        self.add_loss(klLoss, inputs=inputs)

        return inputs


###########
# Encoder #
###########
encoderInput      = Input(shape=(img_rows, img_cols, nChns))

x                 = Conv2D(filters=32, kernel_size=3, activation='relu', strides=2, padding='same')(encoderInput)
x                 = BatchNormalization()(x) # Ensures a steady mean and variance (0,1) to ensure numerical stability

x                 = Conv2D(filters=64, kernel_size=3, activation='relu', strides=2, padding='same')(x)
x                 = BatchNormalization()(x)

convShape         = k_be.int_shape(x)

x                 = Flatten()(x)
x                 = Dense(16, activation='relu')(x)
x                 = BatchNormalization()(x)

latent_space      = Dense(latentDimension, name='z_mean')(x)
z_log_var         = Dense(latentDimension, name='z_log_var')(x)
z_mean, z_log_var = KullbackLeiblerDivergenceLayer()([latent_space, z_log_var])


############################
# Reparameterization trick #
############################
z = Lambda(sampleLatentSpace, output_shape=(latentDimension, ), name='z')([z_mean, z_log_var])

encoder = Model(encoderInput, [z_mean, z_log_var, z], name='encoder')
encoder.summary()


###########
# Decoder #
###########
decoderInput  = Input(shape=(latentDimension,))

x             = Dense(convShape[1] * convShape[2] * convShape[3], activation='relu')(decoderInput)
x             = BatchNormalization()(x)

x             = Reshape((convShape[1], convShape[2], convShape[3]))(x)
x             = Conv2DTranspose(filters=64, kernel_size=3, activation='relu', strides=2, padding='same')(x)
x             = BatchNormalization()(x)

x             = Conv2DTranspose(filters=32, kernel_size=3, activation='relu', strides=2, padding='same')(x)
x             = BatchNormalization()(x)
decoderOutput = Conv2DTranspose(filters=nChns, kernel_size=3, activation='sigmoid', padding='same')(x)

decoder = Model(decoderInput, decoderOutput, name='decoder')
decoder.summary()


############################
# Variational Auto-Encoder #
############################
decoderOutput = decoder(encoder(encoderInput)[2])
VAE           = Model(encoderInput, decoderOutput, name='VAR')
VAE.summary()


######################################
# Compiling Variational Auto-Encoder #
######################################
VAE.compile(optimizer='adam', loss=reconstructionLoss)
tf.keras.utils.plot_model(VAE, to_file='VAE.png', show_shapes=True)


#####################################
# Training Variational Auto-Encoder #
#####################################
history = VAE.fit(x_train, x_train, epochs=epochs, batch_size=batchSize, validation_data=(x_test, x_test))


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
def plotGeneration(decoder, img_rows, img_cols, scale=4.0, n=15):
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
def plotComparisonOriginal(decodedImages, data, n=5):
    for indx in range(n):
        plotIndx = indx * 2 + 1

        plt.subplot(n, 2, plotIndx)
        plt.imshow(data[indx, :, :], cmap='Greys_r')

        plt.subplot(n, 2, plotIndx + 1)
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
encodedImages, _, _ = encoder.predict(x_train)
decodedImages = decoder(encodedImages)
print('Image distance between original and reconstructed:', get_value(reconstructionLoss(x_train[0], decodedImages[0])))
decodedImages = np.reshape(decodedImages, newshape=(-1, img_rows, img_cols))

plotHistory(history)
plotGeneration(decoder, img_rows, img_cols)
plotComparisonOriginal(decodedImages, x_train_orig)
plotLatentSpace(encodedImages, y_train_orig)


###############################
# Auto-Encoder for comparison #
###############################
encoderAE     = Model(encoderInput, latent_space, name='encoderAE')
decoderOutput = decoder(encoderAE(encoderInput))
AE            = Model(encoderInput, decoderOutput, name='AR')
AE.compile(optimizer='adam', loss=reconstructionLoss)
tf.keras.utils.plot_model(AE, to_file='AE.png', show_shapes=True)
historyAE = AE.fit(x_train, x_train, epochs=epochs, batch_size=batchSize, validation_data=(x_test, x_test))


############
# Plotting #
############
encodedImages = encoderAE.predict(x_train)
decodedImages = decoder(encodedImages)
print('Image distance between original and reconstructed:', get_value(reconstructionLoss(x_train[0], decodedImages[0])))
decodedImages = np.reshape(decodedImages, newshape=(-1, img_rows, img_cols))

plotHistory(historyAE)
plotGeneration(decoder, img_rows, img_cols)
plotComparisonOriginal(decodedImages, x_train_orig)
plotLatentSpace(encodedImages, y_train_orig)


print('\n=== DONE ===')
