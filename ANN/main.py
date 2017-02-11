import numpy as np
import os as os
from neuralnet import NeuralNet
from datapoint import dataPoints


# The final list loaded and ready to be sent into the network
trainingData = []
numOfEpochs = 1

def loadData():
    # Listing the directory of the training and testing files
    dirName = os.path.dirname(__file__)
    relPath = "data"
    absPath = os.path.join(dirName, relPath)
    # Lists the directory of data files
    dirListing = os.listdir(absPath)
    # Load the first file so we can concatenate on it
    tempData = np.loadtxt(os.path.join(absPath, dirListing[10]), delimiter=',')
    # Create a array listing all data
    for i in range(9):
        digit_train_list = np.loadtxt(os.path.join(absPath, dirListing[i + 11]), delimiter=',')
        tempData = np.concatenate((tempData, digit_train_list), axis = 0)
    # Used in this for loop as an identifier
    n = 0;
    # For loop used to load all data in individual objects    
    for j in range(tempData.shape[0]):
        dP = dataPoints(tempData[j], n)
        trainingData.append(dP)
        if (j+1)%700 == 0:
            n+=1

  
   
def main():
    np.random.seed(1)
    loadData()
    NN = NeuralNet(1)
    
    for i in range(numOfEpochs):
        for j in range(1):
            print("---------Iteration:---------", j + 1)
            NN.loadNextInputValues(trainingData[j].dataList, trainingData[j].dataId)
            NN.feedForward()
if __name__ == '__main__':
    main()