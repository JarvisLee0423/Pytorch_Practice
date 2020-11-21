#============================================================================================#
#   Copyright:          JarvisLee                       
#   Date:               2020/08/20
#   Project Name:       FizzbuzzNN.py
#   Description:        Building a three layers neural network to resolve the fizzbuzz game
#                       between 0 and upper bound (UB).
#   Model Description:  Hypothesis 1    ->  n_1 nodes
#                       ReLu            ->  Activation Function
#                       Hypothesis 2    ->  n_2 nodes
#                       ReLu            ->  Activation Function
#                       Hypothesis 3    ->  n_3 nodes
#                       softmax         ->  Classifier
#============================================================================================#

# Importing the necessary libraries.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Configuring the device.
if torch.cuda.is_available:
    # Fixing the random seed.
    torch.cuda.manual_seed(1)
    # Fixing the avaliable device.  
    torch.cuda.set_device(0)
    # Applying the GPU.   
    device = 'cuda'             
else:
    # Fixing the random seed.
    torch.manual_seed(1)
    # Applying the CPU.       
    device = 'cpu'              

# Setting up the hyperparameters.
# The number of the total data plus one.
m = 2049
# The number of the features of each data.                        
n_0 = int(np.log2(m - 1))
# The number of the nodes for the first hidden layer.       
n_1 = 100
# The number of the nodes for the second hidden layer.                     
n_2 = 100
# The number of the nodes for the classifier.                       
n_3 = 4
# The value of the learning rate.                        
learning_rate = 0.01
# The value of the epoch.            
epoch = 100
# The value of the batch size.                    
becth_size = 128                

# Defining the class for preparing the data.
class dataGenerator():
    # Defining the fizzbuzz classes.
    @staticmethod
    def fizzbuzz(num):
        # The forth class 'fizz'.
        if num % 3 == 0:
            return 3
        # The third class 'buzz'.        
        elif num % 5 == 0: 
            return 2
        # The second class 'fizzbuzz'.        
        elif num % 15 == 0:
            return 1
        # The first clss 'Null'
        else:
            return 0
    # Generating the data.
    @staticmethod
    def getData(UB):
        # Generating the raw data.
        rawData = [num for num in range(UB + 1)]
        # Getting the raw data.    
        return rawData
    # Initilizating the training data.
    @staticmethod
    def generateData(rawData):
        # The list for training data.
        trainData = []
        # The list for development data.          
        devData = []
        # The list for testing data.            
        testData = []
        # Spliting the whole training data into three parts.           
        for each in rawData:
            # Getting the training data.    
            if torch.rand(1) > 0.1:
                trainData.append(each)
            # Getting the testing data.
            elif torch.rand(1) > 0.95:
                testData.append(each)
            # Getting the development data.
            else:
                devData.append(each)
        # Returning the data.
        return trainData, devData, testData
    # Encoding the data.
    @staticmethod
    def Encoder(data, UB):
        # Getting the binary representation of the data.
        return np.array([data >> i & 1 for i in range(int(np.log2(UB)))])
    # Figuring out the label.
    @staticmethod
    def labelEncoder(data):
        # Getting the label of each number.
        return torch.LongTensor([dataGenerator.fizzbuzz(each) for each in data])
    # Decoding the label.
    @staticmethod
    def labelDecoder(data):
        # Getting the final prediction.
        return ['Null', 'fizzbuzz', 'buzz', 'fizz'][data]

# Creating the model class.
class FizzbuzzNN(nn.Module):
    # Defining the constructor.
    def __init__(self, n_0, n_1, n_2, n_3):
        # Inheritting the super constructor.
        super(FizzbuzzNN, self).__init__()
        # The first layer.
        self.linear_1 = nn.Linear(n_0, n_1)
        # The second layer.
        self.linear_2 = nn.Linear(n_1, n_2)
        # The third layer.
        self.linear_3 = nn.Linear(n_2, n_3)
    # Defining the forward propagation.
    def forward(self, x):
        # Applying the first layer.
        x = self.linear_1(x)
        # Applying the activation.
        x = F.relu(x)
        # Applying the second layer.
        x = self.linear_2(x)
        # Applying the activation.
        x = F.relu(x)
        # Applying the third layer.
        x = self.linear_3(x)
        # Returning the result.
        return x
    # Defining the function for the evaluation.
    @staticmethod
    def evaluator(devSet, devLabelSet, model, loss):
        # Evaluating the model.
        # Doing the prediction.
        prediction = model(devSet)
        # Getting the value of the loss.                                              
        evalCost = loss(prediction, devLabelSet).item()
        # Getting the value of the accuracy.                         
        evalAccuracy = (torch.argmax(prediction, 1) == devLabelSet)
        # Getting the value of the accuracy.             
        evalAccuracy = (evalAccuracy.sum().float() / len(evalAccuracy)).item()
        # Return the cost and accuracy.
        return evalCost, evalAccuracy                                           
    # Defining the function for the training.
    @staticmethod
    def trainer(trainingSet, trainingLabelSet, devSet, devLabelSet, m = m, n_0 = n_0, n_1 = n_1, n_2 = n_2, n_3 = n_3, lr = learning_rate, epoch = epoch, bs = becth_size):
        # Preparing for training.
        # Creating the model.
        model = FizzbuzzNN(n_0, n_1, n_2, n_3).to(device)
        # Setting the loss function. 
        loss = nn.CrossEntropyLoss()
        # Setting the optimizer.                    
        optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = 0.00005)
        # Training the model.
        for epoch in range(epoch):
            # Getting the cost.
            costs = []
            # Getting the accuracy.                                                                                                                  
            accuracies = []
            # Getting the evaluation accuracy.                                                                                                            
            evalAccuracies = []
            # Applying the minibacth gradient descent.
            for start in range(0, len(trainingSet), bs):
                # Getting each minibatch.                                                                               
                trainingData = trainingSet[start : (start + bs)]
                # Getting the corresponding label.                                                                        
                trainingLabel = trainingLabelSet[start : (start + bs)]
                # Doing the prediction.                                                                 
                prediction = model(trainingData)
                # Getting the value of the loss.                                                                                        
                losses = loss(prediction, trainingLabel)
                # Clearing the previous gradient.                                                                                
                optimizer.zero_grad()
                # Applying the backward propagation.                                                                                                   
                losses.backward()
                # Updating the parameters.                                                                                                       
                optimizer.step()
                # Getting the value of cost function.                                                                                                        
                costs.append(losses.item())
                # Getting the value of the accuracy.                                                                                             
                accuracy = (torch.argmax(prediction, 1) == trainingLabel)
                # Getting the value of the accuracy.                                                              
                accuracy = accuracy.sum().float() / len(accuracy)
                # Getting the value of accuracy.                                                                       
                accuracies.append(accuracy.item())
            # Applying the evaluation.                                                                                            
            evalCost, evalAccuracy = FizzbuzzNN.evaluator(devSet, devLabelSet, model.eval(), loss)
            # Figuring out the best model.                                             
            if len(evalAccuracies) == 0 or evalAccuracy > max(evalAccuracies):
                # Saving the model.
                torch.save(model.train().state_dict(), './Simple_Neural_Network_with_Pytorch/Fizzbuzz_Game_Implementation/Fizzbuzz.pt') 
                print("Model Saved")
            # Storing the accuracy of the evaluation for picking up the best model.
            evalAccuracies.append(evalAccuracy)
            # Converting the model mode.
            model.train()                                                                                     
            print("The value of training loss for epoch " + str(epoch + 1) + " is: " + str(np.sum(costs) / len(costs)))
            print("The value of training accuracy for epoch " + str(epoch + 1) + " is: " + str(np.sum(accuracies) / len(accuracies)))
            print("The value of evaluation loss for epoch " + str(epoch + 1) + " is: " + str(evalCost))
            print("The value of evaluation accuracy for epoch " + str(epoch + 1) + " is: " + str(evalAccuracy))

# Training and testing the model.
if __name__ == "__main__":
    cmd = input("Please choose train a model or evaluate the model ('T' for train, 'E' for evaluate, 'Exit' for quit): ")
    while cmd != "Exit":
        if cmd == 'T':
            # Getting the total number of the fizzbuzz game.
            m = input("Please input the upper bound of the fizzbuzz game (Integer Value, 'Exit' for quit)): ")
            while True:
                try:
                    # Setting the total number of the data.
                    m = int(m) + 1
                    # Setting the number of features for each data.
                    n_0 = int(np.log2(m - 1))
                    # Initializating the training data.
                    # Generating all the data.
                    trainData, devData, testData = dataGenerator.generateData(dataGenerator.getData(m - 1))
                    # Getting the training data.                                                 
                    trainingSet = torch.tensor([dataGenerator.Encoder(data, m - 1) for data in trainData], dtype = torch.float32).to(device)
                    # Getting the training label.  
                    trainingLabelSet = dataGenerator.labelEncoder(trainData).to(device)
                    # Getting the development data.                                                       
                    devSet = torch.tensor([dataGenerator.Encoder(data, m - 1) for data in devData], dtype = torch.float32).to(device)
                    # Getting the development label.         
                    devLabelSet = dataGenerator.labelEncoder(devData).to(device)
                    # Getting the testing data.                                                              
                    testSet = torch.tensor([dataGenerator.Encoder(data, m - 1) for data in testData], dtype = torch.float32).to(device)
                    # Getting the testing label.      
                    testLabelSet = dataGenerator.labelEncoder(testData).to(device)
                    # Doing the training.                                                            
                    FizzbuzzNN.trainer(trainingSet, trainingLabelSet, devSet, devLabelSet, m = m, n_0 = n_0)                                 
                    # Evaluating the model.
                    # Creating the evaluating model.  
                    model = FizzbuzzNN(n_0, n_1, n_2, n_3)
                    # Loading the model.                                                                   
                    model.load_state_dict(torch.load('./Simple_Neural_Network_with_Pytorch/Fizzbuzz_Game_Implementation/Fizzbuzz.pt'))
                    # Getting the evaluating result.
                    evalCost, evalAccuracy = FizzbuzzNN.evaluator(testSet, testLabelSet, model.to(device).eval(), loss = nn.CrossEntropyLoss())
                    print("The value of testing loss is: " + str(evalCost))
                    print("The value of testing accuracy is: " + str(evalAccuracy))
                    cmd = input("Now you can evaluate the model ('E' for evaluate, 'Exit' for quit): ")
                    break
                except:
                    if m == "Exit":
                        cmd = m
                        break
                    else:
                        m = input("Invalid Input! Please input the upper bound of the fizzbuzz game (Integer Value, 'Exit' for quit)): ")
        elif cmd == 'E':
            # Getting the corresponding weights' dimensions.
            param = torch.load('./Simple_Neural_Network_with_Pytorch/Fizzbuzz_Game_Implementation/Fizzbuzz.pt')
            # Getting the number of features.
            n_0 = param['linear_1.weight'].shape[1]
            # Getting the number of nodes for first layers.
            n_1 = param['linear_2.weight'].shape[1]
            # Getting the number of nodes for second layers.
            n_2 = param['linear_3.weight'].shape[1]
            # Getting the number of nodes for third layers.
            n_3 = param['linear_3.weight'].shape[0]
            # Generating the model.
            # Creating the evaluating model.
            model = FizzbuzzNN(n_0, n_1, n_2, n_3)                                                                                 
            try:
                # Loading the model.
                model.load_state_dict(torch.load('./Simple_Neural_Network_with_Pytorch/Fizzbuzz_Game_Implementation/Fizzbuzz.pt'))  
                number = input("Please input a integer between " + str(0) + " and " + str(np.power(2, model.state_dict().get("linear_1.weight").shape[1])) + " ('Exit' for quit): ")
                while True:
                    try:
                        number = int(number)
                        if number > np.power(2, model.state_dict().get("linear_1.weight").shape[1]) or number < 0:
                            number = input("Invalid Input! Please input a integer between " + str(0) + " and " + str(np.power(2, model.state_dict().get("linear_1.weight").shape[1])) + " ('Exit' for quit): ")
                        else:
                            # Converting the model into evaluation model.
                            model.eval().to(device)
                            # Converting the input number into the testing data.
                            testData = torch.tensor(dataGenerator.Encoder(number, np.power(2, model.state_dict().get("linear_1.weight").shape[1])), dtype = torch.float32).to(device)
                            # Evaluating the model.
                            testLabel = torch.argmax(model(testData))
                            print("The model predict the input is: " + dataGenerator.labelDecoder(testLabel.item()))
                            print("The real label is: " + dataGenerator.labelDecoder(dataGenerator.fizzbuzz(number)))
                            if dataGenerator.labelDecoder(testLabel) == dataGenerator.labelDecoder(dataGenerator.fizzbuzz(number)):
                                number = input("Successed! Please input a integer between " + str(0) + " and " + str(np.power(2, model.state_dict().get("linear_1.weight").shape[1])) + " ('Exit' for quit): ")
                            else:
                                number = input("Failure! Please input a integer between " + str(0) + " and " + str(np.power(2, model.state_dict().get("linear_1.weight").shape[1])) + " ('Exit' for quit): ")
                    except:
                        if number == "Exit":
                            cmd = number
                            break
                        else:
                            number = input("Invalid Input! Please input a integer between " + str(0) + " and " + str(np.power(2, model.state_dict().get("linear_1.weight").shape[1])) + " ('Exit' for quit): ")
            except:
                print("No model has been found, please train one first!!!")
                cmd = 'T'
        else:
            cmd = input("Invalid command! Please try again ('T' for train, 'E' for evaluate, 'Exit' for quit): ")