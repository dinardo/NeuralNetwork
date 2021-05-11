"""
#############################
Neuron.py by Mauro E. Dinardo
#############################
"""
from random import gauss
from math   import sqrt, exp, log, tanh, atanh
"""
###################################################
- Quadratic cost functions: 1/2 (target - result)^2
  Cross-entropy cost function, i.e.
  dC/dwj = xj * (result - target), it avoids the
  learning slowdown caused by the first derivative
  of the activation function
- Regularization: L2
- RMSprop: implemented

Nvars    = number of input variables
aFunType = type of activation function:
           'tanh', 'sigmoid', 'ReLU', 'lin', 'BPN'
###################################################
"""
class Neuron(object):
    def __init__(self,Nvars,aFunType):
        if aFunType == 'tanh' or aFunType == 'sigmoid' or aFunType == 'ReLU' or aFunType == 'lin' or aFunType == 'BPN':
            self.aFunType = aFunType
        else:
            print('[Neuron::__init__]\tWrong option:', aFunType)
            quit()

        self.Nvars      = Nvars

        ### Hyper-parameters ###
        self.learnRate  =  0.01 # Range [0; infinite)
        self.regular    =  1e-3
        self.rmsPrDecay =  0.9  # Range [0; 1 (= no RMSprop)]
        ########################

        self.aFunMin    = -1.
        self.aFunMax    = +1.
        self.aFunRange  =  3. # Activation function input range for [10% - 90%] output range

        self.rmsProp    =  0.

        self.afun       =  0.
        self.dafundz    =  0.

        self.amIfixed   = False
        self.amIminiB   = False

        self.weights = [ gauss(0,self.aFunRange / sqrt(self.Nvars)) for k in range(self.Nvars) ]
        self.weights.append(gauss(0,self.aFunRange))

        if self.aFunType is 'BPN':
            self.weights = [ 0. for k in range(self.Nvars+1) ]

    ### Return the value of the activation function ###
    def eval(self,invec):
        """
        For a backpropagation network the activation function is equal to dC/dz
        i.e. for a neuron i in layer j the activation function is equal to dC/dz_ij
        """
        wsum  = sum(W * i for W,i in zip(self.weights,invec))
        wsum += self.weights[self.Nvars]

        self.afun    = self.aFun(wsum)
        self.dafundz = self.daFunDz()

        return self.afun

    ### Update the weights ###
    def adapt(self,invec,dCdZ,correct=[]):
        """
        For a feedforward network the product dCdZ * invec[k] is equal to dC/dw_k
        i.e. for a neuron i in layer j dCdZ * invec[k] = dC/w_ijk
        """
        if self.amIfixed == False:
            self.rmsProp = self.rmsPrDecay * self.rmsProp + (1. - self.rmsPrDecay) * dCdZ * dCdZ
            rmsProp_     = self.rmsProp if self.rmsProp > 0 else 1.

            for k in range(self.Nvars+1):
                if not correct:
                    if k == self.Nvars:
                        self.weights[k] = self.learnRate * dCdZ / sqrt(rmsProp_) if self.amIminiB == True else self.weights[self.Nvars] - self.learnRate * dCdZ / sqrt(rmsProp_)
                    else:
                        self.weights[k] = self.learnRate * dCdZ * invec[k] / sqrt(rmsProp_) if self.amIminiB == True else (1. - self.regular*self.learnRate) * self.weights[k] - self.learnRate * dCdZ * invec[k]
                else:
                    if k == self.Nvars:
                        self.weights[k] = correct[k] if self.amIminiB == True else self.weights[self.Nvars] - correct[k]
                    else:
                        self.weights[k] = correct[k] if self.amIminiB == True else (1. - self.regular*self.learnRate) * self.weights[k] - correct[k]

    ### Activation function ###
    def aFun(self,val):
        if self.aFunType == 'tanh':
            return tanh(val)

        if self.aFunType == 'sigmoid':
            return 1. - 2. / (1. + exp(val))

        if self.aFunType == 'ReLU':
            return max(val,0)

        if self.aFunType == 'lin':
            return val

        if self.aFunType == 'BPN':
            return val * self.dafundz

    ### d(Activation function) / dz ###
    def daFunDz(self):
        if self.aFunType == 'tanh':
            return 1. - self.afun * self.afun

        if self.aFunType == 'sigmoid':
            return (1. - self.afun*self.afun) / 2.

        if self.aFunType == 'ReLU':
            return 1. if self.afun > 0 else 0.

        if self.aFunType == 'lin':
            return 1.

        if self.aFunType == 'BPN':
            return 0.

    ### Cost function ###
    def cFun(self,target):
        # @TMP@
#        if self.aFunType == 'tanh':
#            return - 1/2. * log(self.dafundz) - target * atanh(self.afun)

#        if self.aFunType == 'lin':
        return 1/2. * (target - self.afun) * (target - self.afun)

    ### d(Cost function) / dz ###
    def dcFunDz(self,target):
        # @TMP@
#        if self.aFunType == 'tanh':
#            return (self.afun - target)

#        if self.aFunType == 'lin':
        return (self.afun - target) * self.dafundz

    def reset(self):
            self.__init__(self.Nvars,self.aFunType)

    def sum2W(self):
        return sum(W*W for W in self.weights[:-1])

    def scramble(self):
        for k in range(self.Nvars):
            self.weights[k] = self.weights[k] - (self.weights[k] > 1) * gauss(0,self.aFunRange / sqrt(self.Nvars))
        self.weights[self.Nvars] = self.weights[self.Nvars] - (self.weights[self.Nvars] > 1) * gauss(0,self.aFunRange)

    def removeW(self,who):
        self.weights = [ W for k,W in enumerate(self.weights) if k not in who ]
        self.Nvars   = len(self.weights[:]) - 1

    def addW(self,who):
        self.Nvars += len(who[:])
        for k,pos in enumerate(who):
            self.weights.insert(pos+k,gauss(0,self.aFunRange / sqrt(self.Nvars)))

    def copy(self,N,amIminiB):
        N.Nvars      = self.Nvars

        N.learnRate  = self.learnRate
        N.regular    = self.regular
        N.rmsPrDecay = self.rmsPrDecay

        N.aFunMin    = self.aFunMin
        N.aFunMax    = self.aFunMax
        N.aFunRange  = self.aFunRange

        N.rmsProp    = self.rmsProp

        ### Parameters worth saving ###
        N.aFunType   = self.aFunType
        N.afun       = self.afun
        N.dafundz    = self.dafundz

        N.amIfixed   = self.amIfixed
        N.amIminiB   = amIminiB
        ###############################

        if amIminiB == False:
            for k in range(self.Nvars+1):
                N.weights[k] = self.weights[k]

    def printParams(self):
        print('Type =', self.aFunType, '- aFun =', round(self.afun,2), '- d(aFun)/dz =', round(self.dafundz,2), end='')
        print('- learn rate =', self.learnRate, '- L2 regularization =', self.regular, '- RMS propagation decay =', self.rmsPrDecay, end='')
        print('- am I fixed =', self.amIfixed, '- am I mini-batch =', self.amIminiB)

        for k,W in enumerate(self.weights):
            print('            Weight[', k, '] ', round(W,2))

    def save(self,f):
        out = 'Type = {0:10s} aFun = {1:20f} d(aFun)/dz = {2:20f} Am I fixed = {3:} Weights:'.format(self.aFunType,self.afun,self.dafundz,self.amIfixed)
        for W in self.weights:
            out += '{0:20f}'.format(W)
        out += '\n'
        f.write(out)

    def read(self,f):
        str2bool = lambda s: True if s == 'True' else False

        line = f.readline()
        lele = line.split()

        while len(lele) == 0 or (len(lele) > 0 and ('#' in lele[0] or 'Neuron[' not in line)):
            line = f.readline()
            lele = line.split()

        w = [ float(a) for a in lele if a.replace('.','').replace('-','').isdigit() ]

        self.aFunType = next(lele[i+2] for i,a in enumerate(lele) if a == 'Type')
        self.amIfixed = str2bool(next(lele[i+4] for i,a in enumerate(lele) if a == 'Am'))

        w.pop(0)
        self.afun     = w.pop(0)
        self.dafundz  = w.pop(0)
        self.weights  = w
