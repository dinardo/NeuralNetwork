"""
##################################
MiniBatchNN.py by Mauro E. Dinardo
##################################
"""
from NeuralNet import NeuralNet
"""
###################################
NN         = Neural Network
Nminibatch = number of mini-batches
###################################
"""
class MiniBatchNN(object):
    def __init__(self,NN,Nminibatch):
        self.Nminibatch = Nminibatch
        self.NN         = NN
        self.indx       = 0
        self.networks   = [ self.NN.copy(True) for m in xrange(self.Nminibatch) ]

    def learn(self,invec,target):
        self.networks[self.indx].learn(invec,target)
        self.indx += 1
        
        """
        ######
        Update
        ######
        """
        if self.indx == self.Nminibatch:
            self.indx = 0
            for j in xrange(self.NN.Nperceptrons):
                for i in xrange(self.NN.FFperceptrons[j].Nneurons):
                    correct = []
                    for k in xrange(len(self.NN.FFperceptrons[j].neurons[i].weights)):
                        """
                        ######################
                        Compute the correction
                        ######################
                        """
                        avg = 0.
                        for m in self.networks:
                            avg += m.FFperceptrons[j].neurons[i].weights[k]
                        correct.append(avg / self.Nminibatch)

                    """
                    ####################################
                    Perform actual update of original NN
                    ####################################
                    """
                    self.NN.FFperceptrons[j].neurons[i].adapt([],0,correct)

                    """
                    ###########################
                    Copy back to all mini-batch
                    ###########################
                    """
                    for k in xrange(len(self.NN.FFperceptrons[j].neurons[i].weights)):
                        for m in self.networks:
                            m.FFperceptrons[j].neurons[i].weights[k] = self.NN.FFperceptrons[j].neurons[i].weights[k]

            self.NN.eval(invec)
            self.NN.backProp(target)

    def scramble(self,who):
        for MB in self.networks:
            MB.scramble(who)
        
    def fixAllBut(self,who):
        for MB in self.networks:
            MB.fixAllBut(who)

    def release(self,who):
        for MB in self.networks:
            MB.release(who)
        
    def remove(self,who):
        for MB in self.networks:
            MB.remove(who)

    def add(self,who):
        for MB in self.networks:
            MB.add(who)
