"""
Example of Variational Auto-Encoder

Courses on Variational Auto-Encoder
- https://tiao.io/post/tutorial-on-variational-autoencoders-with-a-concise-keras-implementation/
- https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73
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
original_dim       = img_rows * img_cols


###########################
# Specify hyperparameters #
###########################
intermediate_dim   = 256
latent_dim         = 2
batch_size         = 100
epochs             = 10
epsilon_std        = 1.0

print('Train shape:',x_train.shape)
print('Test shape:',x_test.shape)

x_train = x_train.reshape(-1, original_dim) / 255.
x_test  = x_test.reshape(-1, original_dim) / 255.


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
x = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu')(x)

z_mu      = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

z_mu, z_log_var = KLDivergenceLayer()([z_mu, z_log_var])

# Reparametrization trick
z_sigma = Lambda(lambda t: K.exp(.5*t))(z_log_var)

eps   = Input(tensor=K.random_normal(shape=(K.shape(x)[0], latent_dim)))
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
decoder = Sequential([Dense(intermediate_dim, input_dim=latent_dim, activation='relu', name='layer1'),
                      Dense(original_dim, activation='sigmoid', name='layer2')
                      ])
x_pred  = decoder(z)
decoder.summary()


######################
# Training the model #
######################
vae = Model(inputs=[x, eps], outputs=x_pred, name='vae')
vae.compile(optimizer='rmsprop', loss=nll)
tf.keras.utils.plot_model(vae, to_file='VAE.png', show_shapes=True)

print(x_train.shape)

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
