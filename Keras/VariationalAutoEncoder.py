"""
Example of Variational Auto-Encoder

Courses on Variational Auto-Encoder
- https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73

Needed libraries
- see file DeepNN.py
"""

import numpy             as np
import pandas            as pd
import tensorflow        as tf
import matplotlib.pyplot as plt
import keras.layers      as layers
import keras.models      as models
import keras.optimizers  as optimizers
import keras.metrics     as metrics

from keras.datasets import mnist
from keras.backend  import random_normal
from keras.losses   import binary_crossentropy


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
(x_train_orig, y_train), (x_test_orig, _) = mnist.load_data()
img_rows, img_cols = x_train_orig.shape[1:]

print('Train shape:',x_train_orig.shape)
print('Test shape:',x_test_orig.shape)

x_train = np.expand_dims(x_train_orig, -1).astype("float32") / 255.
x_test  = np.expand_dims(x_test_orig, -1) .astype("float32") / 255.


###################
# Hyperparameters #
###################
#inOutDimension        = img_rows * img_cols
#intermediateDimension = 256
latentDimension       = 2
batchSize             = 128
epochs                = 30
#epsilon_std           = 1.0


##################################################################
# Create a sampling layer, uses (z_mean, z_log_var) to sample z, #
# the vector encoding a digit                                    #
##################################################################
class Sampling(layers.Layer):

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch             = tf.shape(z_mean)[0]
        dim               = tf.shape(z_mean)[1]
        epsilon           = random_normal(shape=(batch, dim))

        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


###########
# Encoder #
###########
encoderInput = layers.Input(shape=(img_rows, img_cols, 1))
x            = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoderInput)
x            = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
x            = layers.Flatten()(x)
x            = layers.Dense(16, activation="relu")(x)

z_mean    = layers.Dense(latentDimension, name="z_mean")(x)
z_log_var = layers.Dense(latentDimension, name="z_log_var")(x)
z         = Sampling()([z_mean, z_log_var])

encoder = models.Model(encoderInput, [z_mean, z_log_var, z], name="encoder")
encoder.summary()


###########
# Decoder #
###########
decoderInput  = layers.Input(shape=(latentDimension,))
x             = layers.Dense(7 * 7 * 64, activation="relu")(decoderInput)
x             = layers.Reshape((7, 7, 64))(x)
x             = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
x             = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
decoderOutput = layers.Conv2DTranspose( 1, 3, activation="sigmoid",         padding="same")(x)

decoder = models.Model(decoderInput, decoderOutput, name="decoder")
decoder.summary()


############################
# Variational Auto-Encoder #
############################
class VarAE(models.Model):

    def __init__(self, encoder, decoder, **kwargs):
        super(VarAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction       = self.decoder(z)
            reconstruction_loss  = tf.reduce_mean(tf.reduce_sum(binary_crossentropy(data, reconstruction), axis=(1, 2)))
            kl_loss              = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss              = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss           = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)

        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


############################
# Variational Auto-Encoder #
############################
VAE = VarAE(encoder, decoder)
VAE.compile(optimizer=optimizers.Adam())
tf.keras.utils.plot_model(VAE, to_file='VAE.png', show_shapes=True)
history = VAE.fit(x_train, epochs=epochs, batch_size=batchSize)
#history = VAE.fit(x_train, epochs=epochs, batch_size=batchSize, validation_data=(x_test,))


############
# Plotting #
############
fig, ax = plt.subplots(figsize=(5,5))
hist_df = pd.DataFrame(history.history)
hist_df.plot(ax=ax)
ax.set_ylabel('NELBO')
ax.set_xlabel('# epochs')
ax.set_ylim(.99 * hist_df[1:].values.min(),1.1 * hist_df[1:].values.max())
plt.show()


####################################
# Display a grid of sampled digits #
####################################
def plotLatentSpace(vae, n=30, figsize=15):
    digit_size = 28
    scale      = 1.0
    figure     = np.zeros((digit_size * n, digit_size * n))
    grid_x     = np.linspace(-scale, scale, n)
    grid_y     = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample  = np.array([[xi, yi]])
            x_decoded = vae.decoder.predict(z_sample)
            digit     = x_decoded[0].reshape(digit_size, digit_size)
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit

    plt.figure(figsize=(figsize, figsize))
    start_range    = digit_size // 2
    end_range      = n * digit_size + start_range
    pixel_range    = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)

    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap="Greys_r")
    plt.show()

plotLatentSpace(VAE)


#################################################################
# Display how the latent space clusters different digit classes #
#################################################################
def plotLabelClusters(vae, data, labels):
    z_mean, _, _ = vae.encoder.predict(data)

    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()

plotLabelClusters(VAE, x_train, y_train)





"""
import numpy             as np
import pandas            as pd
import tensorflow        as tf
import matplotlib.pyplot as plt

from scipy.stats    import norm
from keras          import backend as K
from keras.layers   import Input, InputLayer, Dense, Lambda, Layer, Add, Multiply
from keras.models   import Model, Sequential
from keras.datasets import mnist


#####################################################
# Load MNIST data, map gray scale 0-256 to 0-1, and #
# reshape the data to make it a vector              #
#####################################################
(x_train, y_train), (x_test, y_test) = mnist.load_data()
img_rows, img_cols = x_train.shape[1:]

print('Train shape:',x_train.shape)
print('Test shape:',x_test.shape)

x_train = x_train.reshape(-1, img_rows * img_cols) / 255.
x_test  = x_test.reshape(-1, img_rows * img_cols) / 255.


###################
# Hyperparameters #
###################
inOutDimension        = img_rows * img_cols
intermediateDimension = 256
latentDimension       = 2
batchSized            = 100
epochs                = 10
epsilon_std           = 1.0


#######################################
# Negative log likelihood (Bernoulli) #
#######################################
def nll(y_true, y_pred):
    lh = K.tf.distributions.Bernoulli(probs=y_pred)
    return - K.sum(lh.log_prob(y_true), axis=-1)


###############################################################################
# Identity transform layer that adds KL divergence to the final model loss    #
# It represents the KL-divergence as just another layer in the neural network #
# with the inputs equal to the outputs: the means and variances for the       #
# variational auto-encoder                                                    #
###############################################################################
class KLDivergenceLayer(Layer):

    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):
        mu, log_var = inputs
        kl_batch = - .5 * K.sum(1 + log_var - K.square(mu) - K.exp(log_var), axis=-1)

        self.add_loss(K.mean(kl_batch), inputs=inputs)

        return inputs


###########
# Encoder #
###########
x = Input(shape=(inOutDimension,))
h = Dense(intermediate_dim, activation='relu')(x)

z_mu      = Dense(latentDimension)(h)
z_log_var = Dense(latentDimension)(h)

z_mu, z_log_var = KLDivergenceLayer()([z_mu, z_log_var])

# Reparametrization trick
z_sigma = Lambda(lambda t: K.exp(.5*t))(z_log_var)

eps   = Input(tensor=K.random_normal(shape=(K.shape(x)[0], latentDimension)))
z_eps = Multiply()([z_sigma, eps])
z     = Add()([z_mu, z_eps])

###############################################
# This defines the Encoder which takes noise  #
# and input and outputs the latent variable z #
###############################################
encoder = Model(inputs=[x, eps], outputs=z)


###############################################################################
# Decoder: Multy Layer Perceptron, specified as single Keras Sequential Layer #
###############################################################################
decoder = Sequential([Dense(intermediate_dim, input_dim=latentDimension, activation='relu', name='layer1'),
                      Dense(inOutDimension, activation='sigmoid', name='layer2')
                      ])
x_pred  = decoder(z)
decoder.summary()


######################
# Training the model #
######################
vae = Model(inputs=[x, eps], outputs=x_pred, name='vae')
vae.compile(optimizer='rmsprop', loss=nll)
tf.keras.utils.plot_model(vae, to_file='VAE.png', show_shapes=True)


############
# Training #
############
hist = vae.fit(x_train, x_train, shuffle=True, epochs=epochs, batch_size=batch_size, validation_data=(x_test, x_test))


############
# Plotting #
############
fig, ax = plt.subplots(figsize=(5,5))
hist_df = pd.DataFrame(hist.history)
hist_df.plot(ax=ax)
ax.set_ylabel('NELBO')
ax.set_xlabel('# epochs')
ax.set_ylim(.99 * hist_df[1:].values.min(),1.1 * hist_df[1:].values.max())
plt.show()


x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
plt.figure(figsize=(5,5))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test, cmap='nipy_spectral')
plt.colorbar()
plt.savefig('VAE_MNIST_latent.pdf')
plt.show()
"""
