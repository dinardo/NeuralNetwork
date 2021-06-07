"""
##########################
MVA.py by Mauro E. Dinardo
##########################
Check hyper-parameter space:
  * number of perceptrons and neurons
  * activation function: tanh, sigmoid, ReLU, lin
  * number of mini-batches
  * learn rate,  RMSprop, regularization
  * scramble and dropout
  * cost function: quadratic (regression), cross-entropy (classification), softmax
####################################################
MVA implementation with Neural Network
e.g.: python MVA.py -nv 2 -np 4 -nn 2 20 20 1 -sc
      Neural Network with two inputs and one output

ToDo:
- normalizzare variabili di input: mean = 0, RMS = 1
- output layer with sigmoid, to go from 0 to 1, and hidden layers with tanh (?)
- bias in weights
- softmax for linear classifier cs231n.github.io
- plot NN output for signal and background
- plot ROC integral
- plot F-score

To Check: https://agenda.infn.it/event/25855/contributions/133765/attachments/82052/107728/ML_INFN_Hackathon_ANN101_giagu.pdf
- check weight initialization: Gaussian, Uniform
- check implementaion of stocastic gradient descent
- check implementaion of RMSprop
- check if output activation function can be made different: linear (regression), sigmoid (classification), softmax (multi-class classification)
####################################################
"""
from argparse    import ArgumentParser
from random      import seed, random, gauss
from math        import sqrt
from os          import system
from sys         import stdout
from timeit      import default_timer

from ROOT        import gROOT, gStyle, gApplication, TCanvas, TGraph, TH1D, TGaxis, TLegend, TLine

from NeuralNet   import NeuralNet
from MiniBatchNN import MiniBatchNN


def ArgParser():
    """
    ###############
    Argument parser
    ###############
    """
    parser = ArgumentParser()
    parser.add_argument('-in', '--inFile',       dest = 'inFile',       type = str,          help = 'Input neural network',             required=False, default='')
    parser.add_argument('-nv', '--Nvars',        dest = 'Nvars',        type = int,          help = 'Number of variables',              required=False)
    parser.add_argument('-np', '--Nperceptrons', dest = 'Nperceptrons', type = int,          help = 'Number of perceptrons',            required=False)
    parser.add_argument('-nn', '--Nneurons',     dest = 'Nneurons',     type = int,          help = 'Number of neurons per perceptron', required=False, nargs='*')
    parser.add_argument('-sc', '--Scramble',     dest = 'scramble',     action='store_true', help = 'Do scramble',                      required=False)

    options = parser.parse_args()

    if options.inFile:
        print('--> I\'m reading the input file:', options.inFile)

    if options.Nvars:
        print('--> I\'m reading the variable number:', options.Nvars)

    if options.Nperceptrons:
        print('--> I\'m reading the perceptron number:', options.Nperceptrons)

    if options.Nneurons:
        print('--> I\'m reading the neuron number per perceptron:', options.Nneurons)

    if options.scramble:
        print('--> I\'m reading the scramble flag:', options.scramble)

    return options


def SetStyle():
    """
    ###################
    ROOT style settings
    ###################
    """
    gROOT.SetStyle('Plain')
    gROOT.ForceStyle()
    gStyle.SetTextFont(42)

    gStyle.SetOptTitle(0)
    gStyle.SetOptFit(0)
    gStyle.SetOptStat(1111)

    gStyle.SetPadTopMargin(0.08)
    gStyle.SetPadRightMargin(0.08)
    gStyle.SetPadBottomMargin(0.12)
    gStyle.SetPadLeftMargin(0.12)

    gStyle.SetTitleFont(42,'x')
    gStyle.SetTitleFont(42,'y')
    gStyle.SetTitleFont(42,'z')

    gStyle.SetTitleOffset(1.2,'x')
    gStyle.SetTitleOffset(1.2,'y')

    gStyle.SetTitleSize(0.05,'x')
    gStyle.SetTitleSize(0.05,'y')
    gStyle.SetTitleSize(0.05,'z')

    gStyle.SetLabelFont(42,'x')
    gStyle.SetLabelFont(42,'y')
    gStyle.SetLabelFont(42,'z')

    gStyle.SetLabelSize(0.05,'x')
    gStyle.SetLabelSize(0.05,'y')
    gStyle.SetLabelSize(0.05,'z')

    TGaxis.SetMaxDigits(3)
    gStyle.SetStatY(0.9)


"""
############
Main program
############
"""
cmd = ArgParser()


"""
##########################
Neural net: initialization
##########################
"""
seed(0)

if cmd.inFile:
    NN = NeuralNet()
    NN.read(cmd.inFile)
else:
    NN = NeuralNet(cmd.Nvars,cmd.Nperceptrons,cmd.Nneurons)
NN.printParams()


"""
###############
Hyperparameters
###############
"""
Ntraining  = 300000
Nruntest   = 10000
Nminibatch = 8
toScramble = {2:[5]}


"""
###########################################
Read additional hyper-parameter information
###########################################
"""
if cmd.inFile:
    nRunTrainingSt,Nminibatch,toScramble = NN.readHypPar(cmd.inFile)


"""
#####################################
Internal parameters: problem specific
#####################################
"""
epochSpan  = 1000

NNoutMin   = NN.FFperceptrons[NN.Nperceptrons-1].neurons[0].aFunMin
NNoutMax   = NN.FFperceptrons[NN.Nperceptrons-1].neurons[0].aFunMax
NNthr      = (NNoutMin + NNoutMax) / 2.

xRng       = 2.
xOff       = 0.2
yRng       = 2.
yOff       = 0.2

noiseBand  = 0.1
loR        = 0.2
hiR        = 0.5

# Returns whether or not the point is in the signal region
isSignal   = lambda x,y,xOff,yOff,rMin,rMax: True if rMin <= sqrt((x-xOff)*(x-xOff) + (y-yOff)*(y-yOff)) and sqrt((x-xOff)*(x-xOff) + (y-yOff)*(y-yOff)) < rMax else False
# Toss a point on a plane (-xRng/2; +xRng/2) x (-yRng/2; +yRng/2)
xyRndPoint = lambda xRng,yRng: (random() * xRng - xRng/2, random() * yRng - yRng/2)

"""
#########################
Graphics layout and plots
#########################
"""
gROOT.Reset()
SetStyle()

cCost   = TCanvas('cCost',   'NN Cost Function', 0, 0, 700, 500)
cAccu   = TCanvas('cAccu',   'NN Accuracy',      0, 0, 700, 500)
cROC    = TCanvas('cROC',    'NN ROC',           0, 0, 500, 500)
cSpeed  = TCanvas('cSpeed',  'NN Speed',         0, 0, 700, 500)
cNNtrai = TCanvas('cNNtrai', 'NN Training',      0, 0, 700, 500)
cNNtest = TCanvas('cNNtest', 'NN Test',          0, 0, 700, 500)
cNNval  = TCanvas('cNNval',  'NN Values',        0, 0, 700, 500)

graphNNcostTrain = TGraph()
graphNNcostTrain.SetTitle('NN cost function;Epoch [#];Cost Function')
graphNNcostTest = TGraph()
graphNNcostTest.SetTitle('NN cost function;Epoch [#];Cost Function')
graphNNcostTest.SetLineColor(2)

graphNNaccuracyTrain = TGraph()
graphNNaccuracyTrain.SetTitle('NN accuracy;Epoch [#];Accuracy [%]')
graphNNaccuracyTest = TGraph()
graphNNaccuracyTest.SetTitle('NN accuracy;Epoch [#];Accuracy [%]')
graphNNaccuracyTest.SetLineColor(2)

graphNNroc = TGraph()
graphNNroc.SetTitle('Receiver Operating Characteristic;False positive rate;True positive rate')

graphStrai = TGraph()
graphStrai.SetTitle('NN training;x;y')
graphStrai.SetMarkerStyle(20)
graphStrai.SetMarkerSize(0.5)
graphStrai.SetMarkerColor(4)
graphStrai.SetMarkerColorAlpha(4,0.5)

graphBtrai = TGraph()
graphBtrai.SetTitle('NN training;x;y')
graphBtrai.SetMarkerStyle(20)
graphBtrai.SetMarkerSize(0.5)
graphBtrai.SetMarkerColorAlpha(2,0.5)

graphStest = TGraph()
graphStest.SetTitle('NN test;x;y')
graphStest.SetMarkerStyle(20)
graphStest.SetMarkerSize(0.5)
graphStest.SetMarkerColorAlpha(4,0.25)

graphBtest = TGraph()
graphBtest.SetTitle('NN test;x;y')
graphBtest.SetMarkerStyle(20)
graphBtest.SetMarkerSize(0.5)
graphBtest.SetMarkerColorAlpha(2,0.25)

graphEtest = TGraph()
graphEtest.SetTitle('NN test;x;y')
graphEtest.SetMarkerStyle(20)
graphEtest.SetMarkerSize(0.5)
graphEtest.SetMarkerColorAlpha(3,0.25)

histoNNS = TH1D('histoNNS','histoNNS',100,NNoutMin,NNoutMax)
histoNNS.SetTitle('NN signal output;NN test;Entries [#]')
histoNNS.SetLineColor(4)

histoNNB = TH1D('histoNNB','histoNNB',100,NNoutMin,NNoutMax)
histoNNB.SetTitle('NN background output;NN test;Entries [#]')
histoNNB.SetLineColor(2)

histoNNE = TH1D('histoNNE','histoNNE',100,NNoutMin,NNoutMax)
histoNNE.SetTitle('NN error output;NN test;Entries [#]')
histoNNE.SetLineColor(3)

legNNcost = TLegend(0.14, 0.82, 0.26, 0.9, '')
legNNcost.SetTextSize(0.03)
legNNcost.SetFillStyle(1001)

legNNaccuracy = TLegend(0.14, 0.82, 0.26, 0.9, '')
legNNaccuracy.SetTextSize(0.03)
legNNaccuracy.SetFillStyle(1001)

legNNspeed = TLegend(0.93, 0.17, 1.0, 1.0, '')
legNNspeed.SetTextSize(0.03)
legNNspeed.SetFillStyle(1001)

graphNNspeed = []


"""
####################
Neural net: training
####################
"""
print('\n\n=== Training neural network ===')
if cmd.scramble == True and toScramble != None:
    print('--> I will scramble: ', toScramble)

MB                 = MiniBatchNN(NN,Nminibatch)
NNspeed            = [ 0. for j in range(NN.Nperceptrons) ]
NNcostTrain        = 0.
NNcostTest         = 0.
countAccuracyTrain = 0.
countAccuracyTest  = 0.
startClock         = default_timer()

for n in range(1,Ntraining+1):
    """
    ####################
    Neural net: training
    ####################
    """
    x,y = xyRndPoint(xRng,yRng)

    if isSignal(x,y,xOff,yOff,gauss(loR,noiseBand),gauss(hiR,noiseBand)) == True:
        target = NNoutMax
        graphStrai.SetPoint(graphStrai.GetN(),x,y)
    else:
        target = NNoutMin
        graphBtrai.SetPoint(graphBtrai.GetN(),x,y)


    """
    ######################
    Neural net: scrambling
    ######################
    """
    if cmd.scramble == True and toScramble != None:
        MB.release({-1:[]})
        MB.scramble(toScramble)
        MB.fixAllBut(toScramble)


    """
    ####################
    Neural net: learning
    ####################
    """
    MB.learn([x,y],[target])


    if n % Nminibatch == 0:
        """
        ######################################################################
        Neural net: saving activation function speed and cost on training data
        ######################################################################
        """
        NNcostTrain += NN.cFun([target]) + NN.cFunRegularizer()
        NNspeed = [ a + NN.speed(j) for j,a in enumerate(NNspeed) ]
        if n % (epochSpan*Nminibatch) == 0:
            graphNNcostTrain.SetPoint(graphNNcostTrain.GetN(),n,NNcostTrain / epochSpan)
            NNcostTrain = 0.

            for j,a in enumerate(NNspeed):
                if n / (epochSpan*Nminibatch) == 1:
                    graphNNspeed.append(TGraph())
                    leg = 'P:' + str(j)
                    legNNspeed.AddEntry(graphNNspeed[j],leg,'L')
                graphNNspeed[j].SetPoint(graphNNspeed[j].GetN(),n,a / epochSpan)
            NNspeed = [ 0. for j in range(NN.Nperceptrons) ]

            print('--> Accomplished: {0:3.0f} %\r'.format(1. * n / Ntraining * 100.), end='')
            stdout.flush()


        """
        #############################################
        Neural net: evalute accuracy on training data
        #############################################
        """
        NNout = NN.eval([x,y])

        if (NNout[0] > NNthr and isSignal(x,y,xOff,yOff,loR,hiR) == True) or (NNout[0] <= NNthr and isSignal(x,y,xOff,yOff,loR,hiR) == False):
            countAccuracyTrain += 1.

        if n % (epochSpan*Nminibatch) == 0:
            graphNNaccuracyTrain.SetPoint(graphNNaccuracyTrain.GetN(),n,countAccuracyTrain / epochSpan * 100.)
            countAccuracyTrain = 0.


        """
        ##################################################
        Neural net: evalute cost and accuracy on test data
        ##################################################
        """
        if n % (epochSpan*Nminibatch) == 0:
            for it in range(epochSpan):
                x,y   = xyRndPoint(xRng,yRng)
                NNout = NN.eval([x,y])

                if isSignal(x,y,xOff,yOff,gauss(loR,noiseBand),gauss(hiR,noiseBand)) == True:
                    target = NNoutMax
                else:
                    target = NNoutMin

                if (NNout[0] > NNthr and isSignal(x,y,xOff,yOff,loR,hiR) == True) or (NNout[0] <= NNthr and isSignal(x,y,xOff,yOff,loR,hiR) == False):
                    countAccuracyTest += 1.
                NNcostTest += NN.cFun([target]) + NN.cFunRegularizer()

            graphNNcostTest.SetPoint(graphNNcostTest.GetN(),n,NNcostTest / epochSpan)
            NNcostTest = 0.

            graphNNaccuracyTest.SetPoint(graphNNaccuracyTest.GetN(),n,countAccuracyTest / epochSpan * 100.)
            countAccuracyTest = 0.

endClock = default_timer()
print('\n--> Training time:', round(endClock - startClock,2), '[s]')

NN.printParams()
NN.save('NeuralNet.txt')


"""
###########################################
Save additional hyper-parameter information
###########################################
"""
NN.saveHypPar('NeuralNet.txt',Ntraining,Nminibatch,cmd.scramble,toScramble)


"""
#################################
Neural net: test and evaluate ROC
#################################
"""
print('\n\n=== Testing neural network ===')
startClock = default_timer()

countTruePos = 0.
countAllPos  = 0.
countTrueNeg = 0.
countAllNeg  = 0.
testNNthr    = NNoutMin

for n in range(1,Nruntest+1):
    x,y = xyRndPoint(xRng,yRng)

    NNout = NN.eval([x,y])

    if NNout[0] > NNthr and isSignal(x,y,xOff,yOff,loR,hiR) == True:
        graphStest.SetPoint(graphStest.GetN(),x,y)
        histoNNS.Fill(NNout[0])
    elif NNout[0] <= NNthr and isSignal(x,y,xOff,yOff,loR,hiR) == False:
        graphBtest.SetPoint(graphBtest.GetN(),x,y)
        histoNNB.Fill(NNout[0])
    else:
        graphEtest.SetPoint(graphEtest.GetN(),x,y)
        histoNNE.Fill(NNout[0])

    ### Evaluate ROC as a function of the cost function's threshold ###
    if NNout[0] > testNNthr and isSignal(x,y,xOff,yOff,loR,hiR) == True:
        countTruePos += 1.
    elif NNout[0] <= testNNthr and isSignal(x,y,xOff,yOff,loR,hiR) == False:
        countTrueNeg += 1.

    if isSignal(x,y,xOff,yOff,loR,hiR) == True:
        countAllPos += 1.
    else:
        countAllNeg += 1.

    if n % epochSpan == 0 and countAllPos != 0 and countAllNeg != 0:
        graphNNroc.SetPoint(graphNNroc.GetN(),1. - countTrueNeg / countAllNeg,countTruePos / countAllPos)
        countTruePos = 0.
        countAllPos  = 0.
        countTrueNeg = 0.
        countAllNeg  = 0.
        testNNthr   += 1. * n / Nruntest * (NNoutMax - NNoutMin)

endClock = default_timer()
print('--> Testing time:', round(endClock - startClock,2), '[s]')


"""
#########################
Neural net: control plots
#########################
"""
cCost.cd()
graphNNcostTrain.Draw('AL')
graphNNcostTest.Draw('Lsame')
legNNcost.AddEntry(graphNNcostTrain,'Training','L')
legNNcost.AddEntry(graphNNcostTest,'Test','L')
legNNcost.Draw('same')
cCost.Modified()
cCost.Update()

cAccu.cd()
graphNNaccuracyTrain.Draw('AL')
graphNNaccuracyTest.Draw('Lsame')
legNNaccuracy.AddEntry(graphNNaccuracyTrain,'Training','L')
legNNaccuracy.AddEntry(graphNNaccuracyTest,'Test','L')
legNNaccuracy.Draw('same')
cAccu.Modified()
cAccu.Update()

cROC.cd()
cROC.SetGrid()
graphNNroc.Draw('AL')
graphNNroc.GetXaxis().SetRangeUser(0,1)
graphNNroc.GetYaxis().SetRangeUser(0,1)
line = TLine(0,0,1,1)
line.SetLineStyle(2)
line.Draw('same')
cROC.Modified()
cROC.Update()

cSpeed.cd()
if len(graphNNspeed[:]) > 0:
    graphNNspeed[0].Draw('AL')
    graphNNspeed[0].SetTitle('NN activation function speed;Epoch [#];Activation Function Speed')
    graphNNspeed[0].SetLineColor(1)
for k in range(1,len(graphNNspeed[:])):
    graphNNspeed[k].SetLineColor(k+1)
    graphNNspeed[k].Draw('L same')
legNNspeed.Draw('same')
cSpeed.Modified()
cSpeed.Update()

cNNtrai.cd()
cNNtrai.DrawFrame(-xRng/2 - xRng/20,-yRng/2 - xRng/20,+xRng/2 + xRng/20,+yRng/2 + xRng/20)
graphStrai.Draw('P')
graphBtrai.Draw('P same')
cNNtrai.Modified()
cNNtrai.Update()

cNNtest.cd()
cNNtest.DrawFrame(-xRng/2 - xRng/20,-yRng/2 - xRng/20,+xRng/2 + xRng/20,+yRng/2 + xRng/20)
graphStest.Draw('P')
graphBtest.Draw('P same')
graphEtest.Draw('P same')
cNNtest.Modified()
cNNtest.Update()

cNNval.cd()
histoNNS.Draw()
histoNNB.Draw('sames')
histoNNE.Draw('sames')
cNNval.Modified()
cNNval.Update()


"""
########################
Wait for keyborad stroke
########################
"""
system('say \'Neural netowrk optimized\'')
gApplication.Run()
