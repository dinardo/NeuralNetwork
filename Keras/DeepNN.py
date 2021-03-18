"""
Courses on neural networks:
- https://github.com/FNALLPC/machine-learning-hats
- http://neuralnetworksanddeeplearning.com
- https://github.com/khanhnamle1994/neural-nets
- https://adeshpande3.github.io/adeshpande3.github.io/A-Beginner's-Guide-To-Understanding-Convolutional-Neural-Networks/

Needed libraries
- pip install numpy
- pip install pandas
- pip install seaborn
- pip install uproot
- pip install matplotlib
- pip install scikit-learn
- pip install scikit-optimize
- pip install tensorflow (or pip install intel-tensorflow)
- pip install keras
- pip install eli5
"""


import uproot
import eli5
import numpy             as np
import pandas            as pd
import tensorflow        as tf
import seaborn           as sns
import matplotlib.pyplot as plt


#######################################
# Fix random seed for reproducibility #
#######################################
np.random.seed(3)
tf.random.set_seed(3)
plt.rcdefaults()


#############
# Variables #
#############
treeName = 'HZZ4LeptonsAnalysisReduced'
VARS     = ['f_mass4l','f_massjj']
ROOTfile = {}
UPfile   = {}
df       = {}

ROOTfile['bkg'] = 'ntuple_4mu_bkg.root'
ROOTfile['sig'] = 'ntuple_4mu_VV.root'


###################
# Open ROOT files #
###################
UPfile['bkg'] = uproot.open(ROOTfile['bkg'])
UPfile['sig'] = uproot.open(ROOTfile['sig'])

print(UPfile['bkg'][treeName].show())


##############################
# Import as Pandas DataFrame #
##############################
#print(UPfile['bkg'][treeName].pandas.df())
df['bkg'] = UPfile['bkg'][treeName].arrays(library="pd")
df['sig'] = UPfile['sig'][treeName].arrays(library="pd")

print(df['bkg'].iloc[:1])
print(df['bkg'].shape)
print(df['bkg'][VARS].iloc[:1])


##########################
# Convert to numpy array #
##########################
print(df['bkg'].values)
print(df['bkg'].values.shape)

mask = (df['bkg']['f_mass4l'] > 125)
print(mask)
print(df['bkg']['f_mass4l'][mask])

df['bkg'] = UPfile['bkg'][treeName].arrays(library="pd", filter_name=VARS)
df['sig'] = UPfile['sig'][treeName].arrays(library="pd", filter_name=VARS)


####################
# Matplotlib plots #
####################
plt.figure(figsize=(5,5), dpi=100)
plt.xlabel(VARS[0])
bins = np.linspace(80, 140, 100)
df['bkg'][VARS[0]].plot.hist(bins=bins, alpha=1, label='bkg', histtype='step')
df['sig'][VARS[0]].plot.hist(bins=bins, alpha=1, label='sig', histtype='step')
plt.legend(loc='upper right')
plt.xlim(80,140)
plt.show()

plt.figure(figsize=(5,5), dpi=100)
plt.xlabel(VARS[1])
bins=np.linspace(0, 2000, 100)
df['bkg'][VARS[1]].plot.hist(bins=bins, alpha=1, label='bkg', histtype='step')
df['sig'][VARS[1]].plot.hist(bins=bins, alpha=1, label='sig', histtype='step')
plt.legend(loc='upper right')
plt.xlim(0,2000)
plt.show()


######################
# Correlation matrix #
######################
corr = df['sig'].corr()
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values)
plt.title('Correlations')
plt.show()


###########################
# Remove undefined values #
###########################
df['sig']= df['sig'][(df['sig'][VARS[0]] > -999) & (df['sig'][VARS[1]] > -999)]
df['bkg']= df['bkg'][(df['bkg'][VARS[0]] > -999) & (df['bkg'][VARS[1]] > -999)]


#########################
# Add isSignal variable #
#########################
df['sig']['isSignal'] = np.ones(len(df['sig']))
df['bkg']['isSignal'] = np.zeros(len(df['bkg']))


"""
DNN implementation
- Dense (fully-connected) NN layers
- Weights are initialized using a small Gaussian random number
- Output layer contains a single neuron in order to make predictions (it uses the sigmoid activation function in order to produce a probability output in the range of 0 to 1)
- Binary_crossentropy loss function during training (standard loss function for binary classification problems)
- We will optimize the model with the Adam algorithm for stochastic gradient descent and we will collect accuracy metrics while the model is trained
"""

from keras.models     import Sequential, Model
from keras.layers     import Input, Dense, Dropout
from keras.optimizers import Adam

NDIM    = len(VARS)
inputs  = Input(shape=(NDIM,), name='input')
hidden  = Dense(10, name='hidden1', kernel_initializer='normal', activation='relu')(inputs) # activation = [tanh, sigmoid, relu]
hidden  = Dropout(0.5)(hidden)
hidden  = Dense(10, name='hidden2', kernel_initializer='normal', activation='relu')(hidden)
outputs = Dense(1, name='output', kernel_initializer='normal', activation='sigmoid')(hidden)

myModel = Model(inputs=inputs, outputs=outputs)
myModel.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) # optimizer = [rmsprop, adam]
myModel.summary()


"""
Dividing the data into testing and training dataset
- Split the data into two parts (one for training + validation and one for testing)
- Apply "standard scaling" preprocessing, i.e. making the mean = 0 and the RMS = 1 for all input variables (based only on the training/validation dataset)
- Define our early stopping criteria to prevent over-fitting and we will save the model based on the best val_loss
"""

dfAll   = pd.concat([df['sig'],df['bkg']])
dataset = dfAll.values
X       = dataset[:,0:NDIM]
Y       = dataset[:,NDIM]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

from sklearn.preprocessing import StandardScaler
scaler  = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test  = scaler.transform(X_test)

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=10)

from keras.callbacks import ModelCheckpoint
model_checkpoint = ModelCheckpoint('dense_model.h5', monitor='loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', save_freq=1)


############
# Training #
############
history = myModel.fit(X_train, Y_train, epochs=1000, batch_size=1024, verbose=0, callbacks=[early_stopping, model_checkpoint], validation_split=0.25)


"""
Optimization of the hyper-perparameters
- The number of hidden -layers num_hidden-
- The number of nodes in each layer -initial_node-
- The fraction of dropout -dropout-
"""

from skopt       import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt.plots import plot_convergence

space = [Integer(1,      3,      name='hidden_layers'),
         Integer(5,      100,    name='initial_nodes'),
         Real   (0.0,    0.9,    name='dropout'),
         Integer(500,    5000,   name='batch_size'),
         Real   (10**-5, 10**-1, name='learning_rate', prior='log-uniform')]

def buildCustomModel(num_hidden=2, initial_node=50, dropout=0.5):
    inputs = Input(shape=(NDIM,), name='input')
    hidden = None

    for i in range(num_hidden):
        hidden = Dense(int(round(initial_node/np.power(2,i))), activation='relu')(inputs if i == 0 else hidden)
        hidden = Dropout(np.float32(dropout))(hidden)

    outputs = Dense(1, name='output', kernel_initializer='normal', activation='sigmoid')(hidden)
    model   = Model(inputs=inputs, outputs=outputs)

    return model

def train(model, batch_size=1000):
    history  = model.fit(X_train, Y_train, epochs=100, batch_size=batch_size, verbose=0, callbacks=[early_stopping, model_checkpoint], validation_split=0.25)
    best_acc = max(history.history['val_accuracy'])

    return best_acc, history

@use_named_args(space)
def objective(**var):
    print('New configuration: {}'.format(var))
    model = buildCustomModel(num_hidden=var['hidden_layers'], initial_node=var['initial_nodes'], dropout=var['dropout'])
    model.compile(optimizer=Adam(lr=var['learning_rate']), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    best_acc, history = train(model=model, batch_size=var['batch_size'])
    print('Best acc: {}\n'.format(best_acc))

    return -best_acc

res_gp = gp_minimize(objective, space, n_calls=20, random_state=3)

plot_convergence(res_gp)
print('Best parameters: \
\n\tbest_hidden_layers = {} \
\n\tbest_initial_nodes = {} \
\n\tbest_dropout = {} \
\n\tbest_batch_size = {} \
\n\tbest_learning_rate = {}\n'.format(res_gp.x[0],
                                      res_gp.x[1],
                                      res_gp.x[2],
                                      res_gp.x[3],
                                      res_gp.x[4]))
plt.show()


############
# Training #
############
model = buildCustomModel(num_hidden=res_gp.x[0], initial_node=res_gp.x[1], dropout=res_gp.x[2])
model.compile(optimizer=Adam(lr=res_gp.x[4]), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
best_acc, history = train(model=model, batch_size=res_gp.x[3])


#################################################
# Make AUC as a function of a certain parameter #
#################################################
from sklearn.metrics import roc_curve, auc
def AUCscan(res_gp, start, stop):
    roc_auc_test  = []
    roc_auc_train = []
    x             = []

    for i in range(start, stop):
        print('\nScanning parameter value:', i)
        model = buildCustomModel(num_hidden=res_gp.x[0], initial_node=i, dropout=res_gp.x[2])
        model.compile(optimizer=Adam(lr=res_gp.x[4]), loss='binary_crossentropy', metrics=['accuracy'])
        model.summary()
        best_acc, history = train(model=model, batch_size=res_gp.x[3])

        prediction = model.predict(X_test)
        fpr, tpr, thresholds = roc_curve(Y_test, prediction)
        roc_auc_test.extend([auc(fpr, tpr)])

        prediction = model.predict(X_train)
        fpr, tpr, thresholds = roc_curve(Y_train, prediction)
        roc_auc_train.extend([auc(fpr, tpr)])

        x.extend([i])

    plt.figure(figsize=(5,5), dpi=100)
    plt.xlabel('initial nodes')
    plt.ylabel('auc')
    plt.plot(x, roc_auc_test, label='test')
    plt.plot(x, roc_auc_train, label='train')
    plt.legend(loc='lower right')
    plt.show()

#AUCscan(res_gp, 5 , 35)


#####################################
# Variable / feature classification #
#####################################
from sklearn.metrics import fbeta_score, make_scorer
from eli5.sklearn    import PermutationImportance

def myFscore(Y_true, Y_pred, beta=0.5):
    workingPoint = 0.8
    return fbeta_score(Y_true, Y_pred > workingPoint, beta=beta)

permutation = PermutationImportance(model, random_state=3, scoring=make_scorer(myFscore, beta=1)).fit(X_test, Y_test)
df_feature  = eli5.format_as_dataframes(eli5.explain_weights(permutation, feature_names=dfAll.columns.tolist()[:NDIM]))

ax = df_feature['feature_importances'].plot.barh(x='feature', y='weight')
ax.set_title('Feature importance')
ax.set_xlabel('F-score')
ax.set_ylabel('Features')
ax.set_xlim([0, 1])
plt.show()


################
# Plot results #
################
plt.figure(figsize=(15,10), dpi=100)

ax = plt.subplot(2, 2, 1)
ax.plot(history.history['loss'], label='loss')
ax.plot(history.history['val_loss'], label='val_loss')
ax.legend(loc='upper right')
ax.set_xlabel('epoch')
ax.set_ylabel('loss')

ax = plt.subplot(2, 2, 2)
ax.plot(history.history['accuracy'], label='accuracy')
ax.plot(history.history['val_accuracy'], label='val_accuracy')
ax.legend(loc='lower right')
ax.set_xlabel('epoch')
ax.set_ylabel('acc')

prediction = model.predict(X_test)
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(Y_test, prediction)
roc_auc = auc(fpr, tpr)

ax = plt.subplot(2, 2, 3)
ax.plot(fpr, tpr, lw=2, color='cyan', label='auc = %.3f' % (roc_auc))
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', label='random chance')
ax.legend(loc='lower right')
ax.set_title('Receiver Operating Curve')
ax.set_xlabel('false positive rate')
ax.set_ylabel('true positive rate')
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])

dataset        = scaler.transform(df['bkg'].values[:,0:NDIM])
bkg_prediction = model.predict(dataset)
dataset        = scaler.transform(df['sig'].values[:,0:NDIM])
sig_prediction = model.predict(dataset)

ax = plt.subplot(2, 2, 4)
bins = np.linspace(0, 1, 100)
ax.hist(bkg_prediction, bins=bins, alpha=0.8, label='bkg', histtype='bar', color='blue')
ax.hist(sig_prediction, bins=bins, alpha=0.8, label='sig', histtype='bar', color='red')
ax.set_xlabel('NN output')
ax.set_ylabel('a.u.')
ax.set_yscale('log')
ax.legend(loc='upper right')
ax.set_xlim([0, 1])

plt.show()


##########################################################################
# Make a regular 2D grid for the inputs and run prediction at each point #
##########################################################################
myXI, myYI = np.meshgrid(np.linspace(-2, 2, 200), np.linspace(-2, 2, 200))
print(myXI.shape)

myZI = model.predict(np.c_[myXI.ravel(), myYI.ravel()])
myZI = myZI.reshape(myXI.shape)


from matplotlib.colors import ListedColormap
plt.figure(figsize=(15,10), dpi=100)
cm        = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])


################################################################
# Plot contour map of NN output overlaid with test data points #
################################################################
ax = plt.subplot(1, 2, 1)
cont_plot = ax.contourf(myXI, myYI, myZI, cmap=cm, alpha=0.8)
ax.scatter(X_test[:,0], X_test[:,1], c=Y_test, cmap=cm_bright, edgecolors='k')
ax.set_xlim(-2,2)
ax.set_ylim(-2,2)
ax.set_xlabel(VARS[0])
ax.set_ylabel(VARS[1])
plt.colorbar(cont_plot, ax=ax, label='NN output')


#########################################################
# Plot decision boundary overlaid with test data points #
#########################################################
ax = plt.subplot(1, 2, 2)
cont_plot = ax.contourf(myXI, myYI, myZI>0.5, cmap=cm, alpha=0.8)
ax.scatter(X_test[:,0], X_test[:,1], c=Y_test, cmap=cm_bright, edgecolor='k')
ax.set_xlim(-2,2)
ax.set_ylim(-2,2)
ax.set_xlabel(VARS[0])
ax.set_ylabel(VARS[1])
plt.colorbar(cont_plot, ax=ax, label='NN output')

plt.show()


###############################
# Add prediction to ROOT tree #
###############################
"""

Not working yet with uproot 4.0.0

myDF               = UPfile['bkg'][treeName].arrays(library="pd", filter_name=VARS)
dataset            = scaler.transform(myDF.values)
myDF['prediction'] = model.predict(dataset)

with uproot.recreate('test.root') as outputFile:
    outputFile[treeName] = uproot.newtree(dict(zip(myDF.columns,myDF.dtypes)))
    myTuple = {}
    for col, ty in zip(myDF.columns,myDF.dtypes):
        myTuple[col] = np.array(myDF[col], dtype=ty)
    outputFile[treeName].extend(myTuple)

print('Prediction saved into ROOT (TTree) file')
"""
