"""
#################################
Perceptron.py by Mauro E. Dinardo
#################################
"""
from math   import sqrt
from Neuron import Neuron
"""
####################################################
Nneurons = number of neurons of the perceptron
Nvars    = number of input variables for each neuron
aFunType = type of activation function
####################################################
"""
class Perceptron(object):
    def __init__(self,Nneurons,Nvars,aFunType):
        self.Nneurons = Nneurons
        self.neurons  = [ Neuron(Nvars,aFunType) for i in xrange(self.Nneurons) ]

    def eval(self,invec):
        return [ N.eval(invec) for N in self.neurons ]

    def adapt(self,invec,dCdZ):
        for a,N in zip(dCdZ,self.neurons):
            N.adapt(invec,a)

    def cFun(self,target):
        return sum(N.cFun(a) for a,N in zip(target,self.neurons))

    def dcFunDz(self,target):
        return [ N.dcFunDz(a) for a,N in zip(target,self.neurons) ]

    def speed(self):
        return sqrt(sum(N.afun * N.afun for N in self.neurons))

    def reset(self):
        for N in self.neurons:
            N.reset()

    def sum2W(self):
        return sum(N.sum2W() for N in self.neurons)
            
    def scramble(self,who):
        if who[0] == -1:
            who = [ i for i in xrange(self.Nneurons) ]

        for i in who:
            self.neurons[i].scramble()

    def removeW(self,who):
        for N in self.neurons:
            N.removeW(who) if type(who) is list else N.__init__(who,N.aFunType)

    def addW(self,who):
        for N in self.neurons:
            N.addW(who) if type(who) is list else N.__init__(who,N.aFunType)

    def fixAllBut(self,who):
        genExp = (N for (i,N) in enumerate(self.neurons) if i not in who)
        for N in genExp:
            N.amIfixed = True

    def release(self,who):
        if who[0] == -1:
            who = [ i for i in xrange(self.Nneurons) ]

        genExp = (N for (i,N) in enumerate(self.neurons) if i in who)
        for N in genExp:
            N.amIfixed = False

    def removeN(self,who):
        if who[0] == -1:
            who = [ i for i in xrange(self.Nneurons) ]

        self.neurons  = [ N for i,N in enumerate(self.neurons) if i not in who ]
        self.Nneurons = len(self.neurons[:])
        
    def addN(self,who):
        self.Nneurons += len(who[:])
        
        for i,pos in enumerate(who):
            self.neurons.insert(pos+i,Neuron(self.neurons[0].Nvars,self.neurons[0].aFunType))

    def copy(self,P,amIminiB):
        for Nfrom,Nto in zip(self.neurons,P.neurons):
            Nfrom.copy(Nto,amIminiB)

    def printParams(self):
        for i,N in enumerate(self.neurons):
            print '        Neuron[', i, '] -->',
            N.printParams()

    def save(self,f):
        for i,N in enumerate(self.neurons):
            f.write('        Neuron[ {0:d} ] --> '.format(i))
            N.save(f)

    def read(self,f):
        line = f.readline()
        lele = line.split()
        
        while len(lele) == 0 or (len(lele) > 0 and ('#' in lele[0] or 'Perceptron[' not in line)):
            line = f.readline()
            lele = line.split()

        for N in self.neurons:
            N.read(f)
