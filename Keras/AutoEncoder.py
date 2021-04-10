"""
Example of Auto-Encoder

Course on Auto-Encoder
- https://blog.paperspace.com/autoencoder-image-compression-keras/

Needed libraries
- see file DeepNN.py
"""

import numpy             as np
import tensorflow        as tf
import matplotlib.pyplot as plt
import keras.layers      as layers
import keras.models      as models
import keras.optimizers  as optimizers

from keras.datasets import mnist
from keras.backend  import mean, square


#######################################
# Fix random seed for reproducibility #
#######################################
np.random.seed(3)
tf.random.set_seed(3)
plt.rcdefaults()


#####################################################
# Load MNIST data, map gray scale 0-256 to 0-1, and #
# reshape the data to make it a vector              #
#####################################################
(x_train_orig, y_train), (x_test_orig, y_test) = mnist.load_data()
img_rows, img_cols = x_train_orig.shape[1:]

print('Train shape:',x_train_orig.shape)
print('Test shape:',x_test_orig.shape)

x_train = x_train_orig.reshape(-1, img_rows * img_cols) / 255.
x_test  = x_test_orig.reshape(-1, img_rows * img_cols) / 255.


###########################
# Specify hyperparameters #
###########################
inOutDimension        = img_rows * img_cols
intermediateDimension = 256
latentDimension       = 2
batchSize             = 256
epochs                = 20


###########
# Encoder #
###########
encoderInput       = layers.Input     (shape=(inOutDimension),      name='EncoderInput')
encoderDenseLayer1 = layers.Dense     (units=intermediateDimension, name='encoderDdense1')   (encoderInput)
encoderActivLayer1 = layers.LeakyReLU (                             name='encoderLeakyReLu1')(encoderDenseLayer1)
encoderDenseLayer2 = layers.Dense     (units=latentDimension,       name='encoderDense2')    (encoderActivLayer1)
encoderOutput      = layers.LeakyReLU (                             name='encoder_output')   (encoderDenseLayer2)

encoder = models.Model(encoderInput, encoderOutput, name='encoderModel')
encoder.summary()


###########
# Decoder #
###########
decoderInput       = layers.Input     (shape=(latentDimension),     name='decoderInput')
decoderDenseLayer1 = layers.Dense     (units=intermediateDimension, name='decoderDense1')    (decoderInput)
decoderActivLayer1 = layers.LeakyReLU (                             name='decoderLeakyReLu1')(decoderDenseLayer1)
decoderDenseLayer2 = layers.Dense     (units=inOutDimension,        name='decoderDense2')    (decoderActivLayer1)
decoderOutput      = layers.LeakyReLU (                             name='decoderOutput')    (decoderDenseLayer2)

decoder = models.Model(decoderInput, decoderOutput, name='decoderModel')
decoder.summary()


################
# Auto-Encoder #
################
AEinput  = layers.Input(shape=(inOutDimension), name='AEinput')
AElatent = encoder(AEinput)
AEoutput = decoder(AElatent)

AE = models.Model(AEinput, AEoutput, name='AE')
AE.summary()


##################
# AE Compilation #
##################
AE.compile(loss='mse', optimizer=optimizers.Adam(lr=0.0005))
tf.keras.utils.plot_model(AE, to_file='AutoEncoder.png', show_shapes=True)


###############
# Training AE #
###############
AE.fit(x_train, x_train, epochs=epochs, batch_size=batchSize, shuffle=True, validation_data=(x_test, x_test))


############
# Plotting #
############
encodedImages     = encoder.predict(x_train)
decodedImages     = decoder.predict(encodedImages)
decodedImagesOrig = np.reshape(decodedImages, newshape=(decodedImages.shape[0], 28, 28))

images2show = 5
for indx in range(images2show):
    plotIndx = indx * 2 + 1

    plt.subplot(images2show, 2, plotIndx)
    plt.imshow(x_train_orig[indx, :, :], cmap='gray')

    plt.subplot(images2show, 2, plotIndx + 1)
    plt.imshow(decodedImagesOrig[indx, :, :], cmap='gray')

plt.figure()
plt.scatter(encodedImages[:, 0], encodedImages[:, 1], c=y_train)
plt.colorbar()
plt.show()
