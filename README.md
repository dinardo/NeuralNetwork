# NeuralNetwork

Feedforward, fully connected, Neural Network with stocastic gradient descent:
- designed with object oriented paradigm
- implemented with composition
- written in python programming language

Multivariate Aanalysis (`MVA.py`) implementation with Neural Network <br>
Synopsis: `python MVA.py -nv 2 -np 4 -nn 2 20 20 1 -sc` <br>
> i.e. 2 input variable | 4 perceptrons | 2 - 20 - 20 - 1 neurons | scramble

Check hyper-parameter space:
- number of perceptrons and neurons
- activation function: tanh, sigmoid, ReLU, lin
- number of mini-batches
- learn rate,  RMSprop, regularization
- scramble and dropout
- cost function: quadratic (regression), cross-entropy (classification), softmax
