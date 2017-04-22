import numpy as np

class NeuralNet:
    inputLayer = None
    hiddenLayer = None
    outputLayer = None
    classifyDigit = None
    
    '''
    The weight matrices for each layer and the
    weight change value matrices for each layer.
    '''
    
    weightValsItoH = None
    deltaWeightsItoH = None
    
    weightValsHToO = None
    
    deltaWeightsHtoO = None
    
    # Holds the previous weights for momentum and other purposes
    prevDeltaWeightsItoH = None
    prevDeltaWeightsHtoO = None
    
    # Sum of the current batch training runs
    sumDeltaGradientsItoH = None
    sumDeltaGradientsHToO = None
    
    # Previous sums of the gradients when doing batch learning
    prevSumDeltaGradientsItoH = None
    prevSumDeltaGradientsHtoO = None
    
    # The bias values
    biasValuesHiddenL = None
    biasValuesOutputL = None
    
    # Specific numbers of each layer
    numOfHiddenNodes = None
    numOfInputNodes = None
    numOfOutputNodes = None
    
    # The error contribution for each layer
    errContributionsOutput = None
    errContributionsHidden = None
    
    # Used for back prop
    learningRate = None
    momentum = None
    
    # Used for rProp
    rPropDeltaIToH = None
    rPropDeltaHToO = None
    
    # Used for Delta Bar Delta
    learningValsHToO = None
    learningValsIToH = None
    
    # Counts how many samples the NN gets right on each epoch
    accuracyCount = 0
    
    def __init__(self, numOfInputNodes, numOfHiddenNodes, numOfOutputNodes, learningRate, momentum):
        self.numOfHiddenNodes = numOfHiddenNodes
        self.numOfInputNodes = numOfInputNodes
        self.numOfOutputNodes = numOfOutputNodes
        self.learningRate = learningRate
        self.momentum = momentum
        
        # Create the weight matrices for the connections in each layer
        self.initializeNetwork()
        
    # Will create and load the information for the network
    def initializeNetwork(self):
        # Create random values for each hidden node
        self.biasValuesHiddenL = np.random.uniform(-0.5, 0.5,(1, self.numOfHiddenNodes))
        self.biasValuesOutputL = np.random.uniform(-0.5, 0.5,(1, self.numOfOutputNodes))
        
        # Create the weights for each layer connections
        self.weightValsItoH = np.random.uniform(-0.5, 0.5,(self.numOfInputNodes, self.numOfHiddenNodes))
        self.weightValsHToO = np.random.uniform(-0.5, 0.5,(self.numOfHiddenNodes, self.numOfOutputNodes))
         
        # Create the weight change matrices for later
        self.deltaWeightsItoH = np.zeros((self.numOfInputNodes, self.numOfHiddenNodes))
        self.deltaWeightsHtoO = np.zeros((self.numOfHiddenNodes, self.numOfOutputNodes))
        
        # Create gradient change matrices for later
        self.sumDeltaGradientsItoH = np.zeros((self.numOfInputNodes, self.numOfHiddenNodes))
        self.sumDeltaGradientsHToO = np.zeros((self.numOfHiddenNodes, self.numOfOutputNodes))
        
        #To hold the previous gradient changes
        self.prevSumDeltaGradientsItoH = np.zeros((self.numOfInputNodes, self.numOfHiddenNodes))
        self.prevSumDeltaGradientsHtoO = np.zeros((self.numOfHiddenNodes, self.numOfOutputNodes))
        
        # Delta Matrices created for rProp
        self.rPropDeltaIToH = np.full((self.numOfInputNodes, self.numOfHiddenNodes), 0.1)
        self.rPropDeltaHToO = np.full((self.numOfHiddenNodes, self.numOfOutputNodes), 0.1)
        
        # The learning rates for each specific weights for each layer.
        self.learningValsIToH = np.full((self.numOfInputNodes, self.numOfHiddenNodes), 0.0005)
        self.learningValsHToO = np.full((self.numOfHiddenNodes, self.numOfOutputNodes), 0.0005)
        
        
    def loadNextInputValues(self, inputValues, digit):
        # Set the target value for this training set
        self.classifyDigit = digit
        # Our new input layer
        self.inputLayer = inputValues
        # Resize to give 2d dimensions
        self.inputLayer = np.resize(self.inputLayer, (1, self.numOfInputNodes))
        
    # Feed forward pass  
    def feedForward(self):
        self.hiddenLayer = np.dot(self.inputLayer, self.weightValsItoH) 
        self.hiddenLayer = self.activationFunction(self.hiddenLayer + self.biasValuesHiddenL, 0)
        self.outputLayer = np.dot(self.hiddenLayer, self.weightValsHToO)
        self.outputLayer = self.activationFunction(self.outputLayer + self.biasValuesOutputL, 0)
        # Increase the accuracy counter after every feed forward that is correct.
        self.accuracyOfNN()
       
    # Calculates the error in system for each output node for a specific sample    
    def calcErrorInSystem(self):
        print("Output: \n", self.outputLayer)
        # Used to return the error
        errArray = np.empty([1, self.outputLayer.shape[1]])
        print("The digit: ", self.classifyDigit)
        errValue = 0
        for j in range(self.outputLayer.shape[1]):
            # Checks to see if x matches output node position if so make the target = 1 else 0
            if(j == self.classifyDigit):
                target = 1
            else:
                target = 0
            # Calculate the current error for each output node
            curErrorVal = (target - self.outputLayer[0,j])**(2)
            errValue = errValue + curErrorVal
        errValue = ((1/2)*errValue)
        print("Error in system for " + str(self.classifyDigit) +str(":"), errValue)
        return errArray

    
    
    # Calculates the error in each output node
    def calcErrorContributionOutputNodes(self):
        # Create the error contribution array for the output layer
        self.errContributionsOutput = np.empty([1, self.outputLayer.shape[1]])
        # The target for each specific output node
        target = 0
        
        for i in range(self.outputLayer.shape[1]):
            output = self.outputLayer[0, i]
            # Checks to see if x matches output node position if so make the target = 1 else 0
            if(i == self.classifyDigit):
                target = 1
            else:
                target = 0
            # Loads errors in to an array
            self.errContributionsOutput[0,i] = output - target
             
    # Back propagation algorithm
    def backProp(self):
        # Get the error first for each output node
        self.calcErrorContributionOutputNodes()
        # Send output values into derivative of activation function
        deriveOutput = self.activationFunction(self.outputLayer, 0, True)
        # Multiply the derived values with the error
        derivAndErr = deriveOutput * self.errContributionsOutput
        
        transDerivAndErr = np.matrix.transpose(derivAndErr)
        #Multiply with specific hidden layer outputs 
        self.deltaWeightsHtoO = np.dot(transDerivAndErr, self.hiddenLayer)
        
        self.deltaWeightsHtoO = np.matrix.transpose(self.deltaWeightsHtoO)
        # Transpose hidden to output weights matrix
        hiddenToOutputTrans = np.matrix.transpose(self.weightValsHToO)
        # Calculates the error at the hidden layer
        self.errContributionsHidden = np.dot(self.errContributionsOutput, hiddenToOutputTrans)
        # Send the hidden layer output values through the derivative of the activation function
        derivHidden = self.activationFunction(self.hiddenLayer, 0, True)
        # multiply the hidden error for each node and derived output values of hidden layer. 
        deltaHiddenAndErr = derivHidden * self.errContributionsHidden
        
        transDeltaHiddenAndErr = np.matrix.transpose(deltaHiddenAndErr)
        
        self.deltaWeightsItoH = np.dot(transDeltaHiddenAndErr, self.inputLayer)
        
        self.deltaWeightsItoH = np.matrix.transpose(self.deltaWeightsItoH)
        # Add the momentum if they have value. i.e don't add it in the first pass
        if(self.prevDeltaWeightsHtoO is None):
            self.weightValsItoH = self.weightValsItoH - self.learningRate * self.deltaWeightsItoH
            self.weightValsHToO = self.weightValsHToO - self.learningRate * self.deltaWeightsHtoO
        else:
            # Update the weights with momentum
            self.weightValsItoH = self.weightValsItoH - (self.learningRate * self.deltaWeightsItoH + self.momentum * self.prevDeltaWeightsItoH)
            self.weightValsHToO = self.weightValsHToO - (self.learningRate * self.deltaWeightsHtoO + self.momentum * self.prevDeltaWeightsHtoO)
        # Set the momentum matrices for the next pass
        self.prevDeltaWeightsItoH = self.deltaWeightsItoH
        self.prevDeltaWeightsHtoO = self.deltaWeightsHtoO
        # Update the Bias
        self.biasValuesHiddenL = self.biasValuesHiddenL + (self.learningRate * deltaHiddenAndErr)
        self.biasValuesOutputL = self.biasValuesOutputL + (self.learningRate * derivAndErr)
    
    def batchTraining(self):
        # Get the error first for each output node
        self.calcErrorContributionOutputNodes()
        
        # Send output values into derivative of activation function
        deriveOutput = self.activationFunction(self.outputLayer, 0, True)
        
        # Multiply the derived values with the error
        derivAndErr = deriveOutput * self.errContributionsOutput
        
        # Transpose the derivate*Err array for easy dot product
        transDerivAndErr = np.matrix.transpose(derivAndErr)
        
        self.deltaWeightsHtoO = np.dot(transDerivAndErr, self.hiddenLayer)
        
        # Transpose the final result to align the proper delta weights in each cell 
        self.deltaWeightsHtoO = np.matrix.transpose(self.deltaWeightsHtoO)
        
        # Sum up the delta's for the whole batch
        self.sumDeltaGradientsHToO += self.deltaWeightsHtoO
        
        # Transpose HToO weights matrix, so we can calculate errors using dot product
        hiddenToOutputTrans = np.matrix.transpose(self.weightValsHToO)
        
        # Calculates the error at the hidden layer
        self.errContributionsHidden = np.dot(self.errContributionsOutput, hiddenToOutputTrans)
        
        # Send the hidden layer output values through the derivative of the activation function
        derivHidden = self.activationFunction(self.hiddenLayer, 0, True)
        
        # multiply the hidden error for each node and derived output values of hidden layer. 
        deltaHiddenAndErr = derivHidden * self.errContributionsHidden
        
        transDeltaHiddenAndErr = np.matrix.transpose(deltaHiddenAndErr)
        
        self.deltaWeightsItoH = np.dot(transDeltaHiddenAndErr, self.inputLayer)
        
        self.deltaWeightsItoH = np.matrix.transpose(self.deltaWeightsItoH)
        
        # Sum up the delta's for weights for ItoH
        self.sumDeltaGradientsItoH += self.deltaWeightsItoH
    
    def deltaBarDelta(self):
        # The decay factor and growth amount k
        d = 0.20
        k = 0.0001
        
        # Get the sign values of each current gradient
        signDIToH = np.sign(self.sumDeltaGradientsItoH)
        signDHToO = np.sign(self.sumDeltaGradientsHToO)
        
        signPrevDIToH = np.sign(self.prevSumDeltaGradientsItoH)
        signPrevDHToO = np.sign(self.prevSumDeltaGradientsHtoO)
            
        checkArrIToH = signDIToH * signPrevDIToH
        checkArrHToO = signDHToO * signPrevDHToO
        
        # Only updates the hidden to output layer weights
        for i in range(self.weightValsHToO.shape[0]):
            for j in range(self.weightValsHToO.shape[1]):
                if(checkArrHToO[i, j] == 1):
                    # Add to the learning rate the k growth factor.
                    self.learningValsHToO[i, j] = self.learningValsHToO[i, j] + k
                    # Multiply the current sum of the gradient.
                    self.sumDeltaGradientsHToO[i, j] = self.sumDeltaGradientsHToO[i, j] * self.learningValsHToO[i, j]
                    # Subtract from the current weight
                    self.weightValsHToO[i, j] = self.weightValsHToO[i, j]  - self.sumDeltaGradientsHToO[i, j]
                
                elif(checkArrHToO[i, j] == -1):
                    # Multiply to the learning rate the decay.
                    self.learningValsHToO[i, j] = self.learningValsHToO[i, j] * (1-d)
                    # Multiply the current sum of the gradient.
                    self.sumDeltaGradientsHToO[i, j] = self.sumDeltaGradientsHToO[i, j] * self.learningValsHToO[i, j]
                    # Subtract from the current weight
                    self.weightValsHToO[i, j] = self.weightValsHToO[i, j] - self.sumDeltaGradientsHToO[i, j]
                else:
                    # Multiply the current sum of the gradient.
                    self.sumDeltaGradientsHToO[i, j] = self.sumDeltaGradientsHToO[i, j] * self.learningValsHToO[i, j]
                    # Subtract from the current weight
                    self.weightValsHToO[i, j] = self.weightValsHToO[i, j]  - self.sumDeltaGradientsHToO[i, j]
        
        # Limit the learning rates
        self.learningValsHToO = np.clip(self.learningValsHToO, 0.0001, 0.005)
        
        # Updates the Input to Hidden layer weights
        for i in range(self.weightValsItoH.shape[0]):
            for j in range(self.weightValsItoH.shape[1]):
                if(checkArrIToH[i, j] == 1):
                    # Add to the learning rate the k growth factor.
                    self.learningValsIToH[i, j] = self.learningValsIToH[i, j] + k
                    # Multiply the current sum of the gradient.
                    self.sumDeltaGradientsItoH[i, j] = self.sumDeltaGradientsItoH[i, j] * self.learningValsIToH[i, j]
                    # Subtract from the current weight
                    self.weightValsItoH[i, j] = self.weightValsItoH[i, j]  - self.sumDeltaGradientsItoH[i, j]
                elif(checkArrIToH[i, j] == -1):
                    # Multiply to the learning rate the decay.
                    self.learningValsIToH[i, j] = self.learningValsIToH[i, j] * (1-d)
                    # Multiply the current sum of the gradient.
                    self.sumDeltaGradientsItoH[i, j] = self.sumDeltaGradientsItoH[i, j] * self.learningValsIToH[i, j]
                    # Subtract from the current weight
                    self.weightValsItoH[i, j] = self.weightValsItoH[i, j]  - self.sumDeltaGradientsItoH[i, j]
                else:
                    # Multiply the current sum of the gradient.
                    self.sumDeltaGradientsItoH[i, j] = self.sumDeltaGradientsItoH[i, j] * self.learningValsIToH[i, j]
                    # Subtract from the current weight
                    self.weightValsItoH[i, j] = self.weightValsItoH[i, j]  - self.sumDeltaGradientsItoH[i, j]
        
        # Limit the learning rates
        self.learningValsIToH = np.clip(self.learningValsIToH, 0.0001, 0.005)
                    
    def rProp(self):
        npos = 1.2
        nneg = 0.5
        
        # Get the sign values of each current and previous delta for each connection matrix
        signDIToH = np.sign(self.sumDeltaGradientsItoH)
        signPrevDIToH = np.sign(self.prevSumDeltaGradientsItoH)
        
        signDHToO = np.sign(self.sumDeltaGradientsHToO)
        signPrevDHToO = np.sign(self.prevSumDeltaGradientsHtoO)
        
        checkArrIToH = signDIToH * signPrevDIToH
        checkArrHToO = signDHToO * signPrevDHToO
        
        # Only updates the hidden to output layer weights
        for i in range(self.weightValsHToO.shape[0]):
            for j in range(self.weightValsHToO.shape[1]):
                if(checkArrHToO[i,j] == 1):
                    # Update the delta, before applying it
                    self.rPropDeltaHToO[i, j] = self.rPropDeltaHToO[i,j] * npos
                    
                    # Decide what we need to do with delta
                    if(self.sumDeltaGradientsHToO[i, j] > 0):
                        
                        self.weightValsHToO[i, j] = self.weightValsHToO[i,j] - self.rPropDeltaHToO[i, j]
                    
                    elif(self.sumDeltaGradientsHToO[i, j] < 0):
                        
                        self.weightValsHToO[i, j] = self.weightValsHToO[i,j] + self.rPropDeltaHToO[i, j]
                    
                    else:
                        self.weightValsHToO[i, j] = self.weightValsHToO[i,j]
                    
                elif(checkArrHToO[i,j] == -1):
                    self.rPropDeltaHToO[i, j] = self.rPropDeltaHToO[i,j] * nneg
                    self.sumDeltaGradientsHToO[i, j] = 0
                    
                else:
                    if(self.sumDeltaGradientsHToO[i, j] > 0):
                        
                        self.weightValsHToO[i, j] = self.weightValsHToO[i, j] - self.rPropDeltaHToO[i, j]
                    
                    elif(self.sumDeltaGradientsHToO[i, j] < 0):
                        
                        self.weightValsHToO[i, j] = self.weightValsHToO[i, j] + self.rPropDeltaHToO[i, j]
                    
                    else:
                        self.weightValsHToO[i, j] = self.weightValsHToO[i,j]
                        
        # Keep the delta's within specific range
        self.rPropDeltaHToO = np.clip(self.rPropDeltaHToO,0.000001, 50)
        
        # Updates the Input to Hidden layer weights
        for i in range(self.weightValsItoH.shape[0]):
            for j in range(self.weightValsItoH.shape[1]):
                if(checkArrIToH[i, j] == 1):
                    self.rPropDeltaIToH[i, j] =  self.rPropDeltaIToH[i, j] * npos
                    
                    if(self.sumDeltaGradientsItoH[i, j] > 0):
                       
                        self.weightValsItoH[i, j] = self.weightValsItoH[i, j] - self.rPropDeltaIToH[i, j]
                    
                    elif(self.sumDeltaGradientsItoH[i, j] < 0):
                        
                        self.weightValsItoH[i, j] = self.weightValsItoH[i, j] + self.rPropDeltaIToH[i, j]
                    
                    else:
                        self.weightValsItoH[i, j] = self.weightValsItoH[i, j]
                elif(checkArrIToH[i, j] == -1):
                    
                    self.rPropDeltaIToH[i, j] = self.rPropDeltaIToH[i, j] * nneg
                    self.sumDeltaGradientsItoH[i, j] = 0
                
                else:
                    
                    if(self.sumDeltaGradientsItoH[i, j] > 0):
                        
                        self.weightValsItoH[i, j] = self.weightValsItoH[i, j] - self.rPropDeltaIToH[i, j]
                    
                    elif(self.sumDeltaGradientsItoH[i, j] < 0):
                       
                        self.weightValsItoH[i, j] = self.weightValsItoH[i, j] + self.rPropDeltaIToH[i, j]
                    
                    else:
                        self.weightValsItoH[i, j] = self.weightValsItoH[i, j]
        
        self.rPropDeltaIToH = np.clip(self.rPropDeltaIToH,0.000001, 50)
    
    # Used to remember the previous batch gradients
    def setSumOfPrevGradients(self):
        # Copy the current gradients over
        self.prevSumDeltaGradientsItoH = np.copy(self.sumDeltaGradientsItoH)
        self.prevSumDeltaGradientsHtoO = np.copy(self.sumDeltaGradientsHToO)
        
        # Clear the current sums for the next batch.
        self.sumDeltaGradientsItoH.fill(0)
        self.sumDeltaGradientsHToO.fill(0)
    
    # Increments a counter every time a sample is classified correctly
    def accuracyOfNN(self):
        # Return the max value from the array
        maxVal = np.amax(self.outputLayer) 
    
        if(maxVal > 0.5 and self.outputLayer[0, self.classifyDigit] == maxVal):
            self.accuracyCount += 1
        
    # Return the accuracy of the network  
    def accuracyOfEpoch(self, numOfSamples):
        correct  = self.accuracyCount
        # Reset the count
        self.accuracyCount = 0
        return (correct/numOfSamples)
    
    def clearAccuracyCount(self):
        self.accuracyCount = 0
    
    # Both activation functions    
    def activationFunction(self, x, func ,derive = False):
        if(func == 0):
            if(derive == True):
                return x*(1-x)
                
            return 1/(1+np.exp(-x))
        else:
            if(derive == True):
                return 1 - (np.tanh(x)) ** 2
                
            return np.tanh(x)
    