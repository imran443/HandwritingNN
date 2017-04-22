# Imran Qureshi, 5510631, 
import numpy as np
import os as os
import csv
from neuralnet import NeuralNet
from datapoint import dataPoints

# Stop scientific notation in numpy arrays.
np.set_printoptions(suppress=True)

# Seed used for testing purposes
#np.random.seed(1)

# Used as place holders
trainingDataTemp = []
testingDataTemp = []

# K Folds
k_foldValue = 10

# Network Configurations
numOfEpochs = 300
numOfInputNodes = 64
numOfHiddenNodes = 25
numOfOutputNodes = 10
learningRate = 0.3
momentum = 0.2

# Create the network
NN = None

# Used to help check error in the system
digits = np.arange(10)

# Used for to hold the error and accuracy returned from the system
errArray = None
accuracyPerEpoch = None

# The .CSV file to write to for each run
file = open("results.csv", 'w', newline='')
wr = csv.writer(file, quoting=csv.QUOTE_ALL)

file2 = open("epochs.csv", 'w', newline='')
wr2 = csv.writer(file2, quoting=csv.QUOTE_ALL)

printList = [[] for i in range(2)]
    

def loadData():
    # Listing the directory of the training and testing files
    dirName = os.path.dirname(__file__)
    relPath = "data"
    absPath = os.path.join(dirName, relPath)
    
    # Lists the directory of data files
    dirListing = os.listdir(absPath)

    # Load the first file so we can concatenate on it for the training samples
    tempData = np.loadtxt(os.path.join(absPath, dirListing[10]), delimiter=',')
      
    # Load the first file so we can concatenate on it for the testing sets
    tempDataTest = np.loadtxt(os.path.join(absPath, dirListing[0]), delimiter=',')
      
    for i in range(9):
        # Concatenate the training data together
        digit_train_list = np.loadtxt(os.path.join(absPath, dirListing[i + 11]), delimiter=',')
        tempData = np.concatenate((tempData, digit_train_list), axis = 0)
        
        # Concatenate the test data together
        digit_test_list = np.loadtxt(os.path.join(absPath, dirListing[i + 1]), delimiter=',')
        tempDataTest = np.concatenate((tempDataTest, digit_test_list), axis = 0)
     
    # Used in this for loop as an identifier
    n = 0;
    
    # For loop used to load all data in individual objects    
    for j in range(tempData.shape[0]):
        dP = dataPoints(tempData[j], n)
        trainingDataTemp.append(dP)
        if ((j + 1) % 700 == 0):
            n+=1
    # Used in this for loop as an identifier
    x = 0;
    
    # For loop used to load all data in individual objects    
    for j in range(tempDataTest.shape[0]):
        dP = dataPoints(tempDataTest[j], x)
        testingDataTemp.append(dP)
        if ((j + 1) % 400 == 0):
            x+=1

'''
    Choose trainType which can be 1 (backProp), 2 (delta bar delta), 
    or 3 (rProp).
    batchT is for batch training. 1 (enabled) and or 0 (disabled).
'''
def holdOut(trainType, batchT):
    # Keeps count of same accuracy occurring
    counter = 0
    
    # Get the training data
    trainingData = np.array(trainingDataTemp)
    
    print("-------- Epoch --------:", 0)
    # Send data for initial run
    training(trainingData, 0, 0)
    # After each epoch calculate the error in the system for each digit
    NN.calcErrorInSystem()
    
    # Get the accuracy of this run
    accuracyPerEpoch = NN.accuracyOfEpoch(trainingData.shape[0])  
    
    print("Accuracy of Epoch 0: " , accuracyPerEpoch)
    
    for i in range(numOfEpochs):
        np.random.shuffle(trainingData)
        # Train the network
        training(trainingData, trainType, batchT)
        
        print("-------- Epoch --------:", i + 1)
        
        # After each epoch calculate the error in the system for each digit
        NN.calcErrorInSystem()
        
        # Get the accuracy of this run
        accuracyPerEpoch = NN.accuracyOfEpoch(trainingData.shape[0])
        
        print("Accuracy of Epoch " + str(i + 1) + str(": "), accuracyPerEpoch)
        
        '''
            If the accuracy is 95% or higher for more than 3 
            epochs straight then system has converged. Else reset the 
            counter.
        '''
        # Stops the run if we reach the required accuracy.
        if(accuracyPerEpoch > 0.95 or (i+1) == numOfEpochs):
            counter +=1
            # Counter is used as indicator that there is no change for 10 epochs.
            if(counter == 3 or (i+1) == numOfEpochs):
                print("---------Training Complete--------")
                # Run the testing data before finishing
                testing()
                wr2.writerow([i + 1])
                return
        else:
            counter = 0

# Use the k-fold method for training
def k_fold(fold, trainType, batchT):
    # Get the data
    trainingData = np.array(trainingDataTemp)
    # Shuffle the whole data set before splitting
    np.random.shuffle(trainingData)
    
    for i in range(numOfEpochs):
        # Sum all the accuracy values of each validation
        sumOfKRuns = 0
        for j in range(fold):
            # Split the data into k subsets
            splitTrainingData = np.split(trainingData, fold)
            validationSet = splitTrainingData[j]
            
            # Remove the validation set from original data
            splitTrainingData.pop(j)
            
            # Convert from Python list to ndarray
            tempTrainingSet = np.array(splitTrainingData)
            trainingSet = np.ndarray.flatten(tempTrainingSet)
            
            # Shuffle both the validation and training sets
            np.random.shuffle(trainingSet) 
            np.random.shuffle(validationSet)
            
            # Perform training for 9/10 training set
            training(trainingSet, trainType, batchT)
            NN.clearAccuracyCount()
            
            # Perform accuracy test for 1/10 validation set, no back prop needed or batch training
            training(validationSet, 0, 0)
            sumOfKRuns += NN.accuracyOfEpoch(validationSet.shape[0])
            
        # Get the average of the sumOfKRuns
        avgOfKRuns = sumOfKRuns/fold
        
        print("-------- Epoch --------:", i + 1)
        print("Average accuracy for K runs: \n", avgOfKRuns)
        
        # If the average percentage is >= 98 % then stop training and validation
        if(avgOfKRuns >= 0.95):
            print("Training Complete: \n", avgOfKRuns)
            # avgOfKRuns printed to .csv
            #wr.writerow([avgOfKRuns])
            testing()
            # Number of epochs
            wr2.writerow([(i + 1) * 10])
            return
 
# Used to train the network, can choose to enable backProp, quickProp, or rProp        
def training(trainingData, trainType, batchT):
    for j in range(trainingData.shape[0]):
        NN.loadNextInputValues(trainingData[j].dataList, trainingData[j].dataId)
        NN.feedForward()
        if(trainType == 1):
            NN.backProp()
        if(batchT == 1):
            NN.batchTraining()
    if(trainType == 2):
        NN.deltaBarDelta()
        # Important step for deltaBarDelta, does not effect vanilla back prop
        NN.setSumOfPrevGradients()
    if(trainType == 3):
        NN.rProp()
        NN.setSumOfPrevGradients()
        
# Tests the network on the testing data after training        
def testing():  
    # Load data
    testingData = np.array(testingDataTemp) 
    np.random.shuffle(testingData)
    
    for j in range(testingData.shape[0]):
        NN.loadNextInputValues(testingData[j].dataList, testingData[j].dataId)
        NN.feedForward()
        
    accuracy = NN.accuracyOfEpoch(testingData.shape[0])
    print("Accuracy of Epoch Of Test Set: ", accuracy)
    # Write to the .CSV file
    wr.writerow([accuracy])

def main():
    global NN
    loadData()
    
    # Normal hold out 
    for i in range(15):
        
        # Create the network
        NN = NeuralNet(numOfInputNodes, numOfHiddenNodes, numOfOutputNodes, learningRate, momentum)
        # Perform either training technique by commenting one of them out.
        #k_fold(k_foldValue, 1, 0)
        holdOut(1, 0)
    # Prints results to .CSV file   
    wr.writerow([ ])
    wr.writerow(["Number of hidden nodes: ", numOfHiddenNodes])
    wr.writerow(["Learning rate: ", NN.learningRate])
    wr.writerow(["Momentum: ", NN.momentum])

    
        
if __name__ == '__main__':
    main()