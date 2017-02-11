import numpy as np

class NeuralNet:
    inputLayer = None
    hiddenLayer = None
    outputLayer = None
    
    target = None
    
    weightValsItoH = None
    weightValsHToO = None
    # Used if second hidden layer is needed
    weightValsHtoH = None
    
    biasValuesHiddenL = None
    biasValuesOutputL = None
    
    numOfHiddenLayers = None # Take input for this value later from user.
    numOfInputLayers  = 1
    numOfOutputLayers = 1
    
    numOfHiddenNodes = None
    numOfInputNodes = 64
    numOfOutputNodes = 1
    
    learningRate = 0.5 # Starting with this value, may need to be adjust it.
    
    def __init__(self, numOfHiddenNodes):
        self.numOfHiddenNodes = numOfHiddenNodes
        self.initializeNetwork()
        
    
    # Will create and load the information for the network
    def initializeNetwork(self):
        # Create random values for each hidden node
        self.biasValuesHiddenL = np.random.uniform(-0.5,0.5,(1, self.numOfHiddenNodes))
        self.biasValuesOutputL = np.random.uniform(-0.5,0.5,(1, self.numOfOutputNodes))
        
        # Create the weights for each layer connections
        self.weightValsItoH = np.random.uniform(-0.5,0.5,(self.numOfInputNodes, self.numOfHiddenNodes))
        self.weightValsHToO = np.random.uniform(-0.5,0.5,(self.numOfHiddenNodes, self.numOfOutputNodes)) 
    
    def loadNextInputValues(self, inputValues, target):
        self.inputLayer = inputValues
        self.inputLayer.resize((1,64))
        print(self.inputLayer.shape)
    
    # TODO!    
    def feedForward(self):
        self.hiddenLayer = np.dot(self.inputLayer, self.weightValsItoH) 
        
        
        print("Input Layer: \n", self.inputLayer)
        print("Bias for Hidden Layer: \n", self.biasValuesHiddenL)
        print("Input Layer to Hidden Layer weights: \n", self.weightValsItoH)
        print("Hidden Layer: \n", self.hiddenLayer)
        self.hiddenLayer = self.sigmoid(self.hiddenLayer + self.biasValuesHiddenL)
        print("Hidden Layer after Sigmoid: \n", self.hiddenLayer)
        print("Hidden Layer to Output Layer Weights: \n", self.weightValsHToO)
        self.outputLayer = np.dot(self.hiddenLayer, self.weightValsHToO)
        print("Bias for Output Layer: \n", self.biasValuesOutputL)
        print("Output Layer: \n", self.outputLayer)
        self.outputLayer = self.sigmoid(self.outputLayer + self.biasValuesOutputL)
        print("Output: \n", self.outputLayer)
        
    
    # TODO!
    def backProp(self):
        return
    
    def sigmoid(self, x, derive=False):
        if(derive==True):
            return x*(1-x)
            
        return 1/(1+np.exp(-x))
    
    