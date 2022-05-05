# NeuralNetwork

Feedforward, fully connected, Neural Network with stocastic gradient descent:
- designed with object oriented paradigm
- implemented with composition
- written in python programming language

Multivariate Analysis (`MVA.py`) implementation with Neural Network <br>
**Synopsis:** `python MVA.py -nv 2 -np 4 -nn 2 20 20 1 -sc` <br>
> i.e. 2 input variable | 4 perceptrons | 2 - 20 - 20 - 1 neurons | scramble

**Check hyper-parameter space before running the program:**
- number of perceptrons and neurons
- activation function: tanh, sigmoid, ReLU, lin
- number of mini-batches
- learn rate,  RMSprop, regularization
- scramble and dropout
- cost function: quadratic (regression), cross-entropy (classification), softmax

**ToDo:**
- normalize input variabile: mean = 0, RMS = 1
- output layer with sigmoid, to go from 0 to 1, and hidden layers with tanh (?)
- bias in weights
- softmax for linear classifier cs231n.github.io
- plot NN output for signal and background
- plot ROC integral
- plot F-score

**To Check: https://agenda.infn.it/event/25855/contributions/133765/attachments/82052/107728/ML_INFN_Hackathon_ANN101_giagu.pdf**
- weight initialization: Gaussian, Uniform
- implementaion of stocastic gradient descent
- implementaion of RMSprop
- if output activation function can be made different: linear (regression), sigmoid (classification), softmax (multi-class classification)
