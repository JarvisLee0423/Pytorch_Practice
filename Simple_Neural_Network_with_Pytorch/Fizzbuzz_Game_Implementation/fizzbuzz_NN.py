#============================================================================================#
#   Copyright:          JarvisLee                       
#   Date:               2020/08/20
#   Project Name:       fizzbuzz_NN.py
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
    torch.cuda.manual_seed(1)   # Fixing the random seed.
    torch.cuda.set_device(0)    # Fixing the avaliable device.
    device = 'cuda'             # Applying the GPU.
else:
    torch.manual_seed(1)        # Fixing the random seed.
    device = 'cpu'              # Applying the CPU.

# Setting up the hyperparameters.
m = 2049
n_0 = int(np.log2(m - 1))
n_1 = 100
n_2 = 100
n_3 = 4
learning_rate = 0.01
epoch = 100
becth_size = 128

# Defining the fizzbuzz game whose target is to figure out whether the number is the multiple of the 3, 5 and 15.
def fizzbuzz(num):
    if num % 3 == 0:
        return 3
    elif num % 5 == 0:
        return 2
    elif num % 15 == 0:
        return 1
    else:
        return 0

# Generating the data.
def getData(UB):
    rawData = [num for num in range(UB + 1)]    # Generating the raw data.
    return rawData                              # Getting the raw data.

# Initilizating the training data.
def generateData(rawData):
    trainData = []          # The list for training data.
    devData = []            # The list for development data.
    testData = []           # The list for testing data.
    for each in rawData:    # Spliting the whole training data into three parts.
        if torch.rand(1) > 0.1:
            trainData.append(each)
        elif torch.rand(1) > 0.95:
            testData.append(each)
        else:
            devData.append(each)
    return trainData, devData, testData

# Encoding the data.
def Encoder(data, UB):
    return np.array([data >> i & 1 for i in range(int(np.log2(UB)))])   # Getting the binary representation of the data.

# Figuring out the label.
def labelEncoder(data):
    return torch.LongTensor([fizzbuzz(each) for each in data])          # Getting the label of each number.

# Decoding the label.
def labelDecoder(data):
    return ['Null', 'fizzbuzz', 'buzz', 'fizz'][data]  # Getting the final prediction.

# Creating the model class.
class fizzbuzz_NN(nn.Module):
    # Defining the constructor.
    def __init__(self, n_0, n_1, n_2, n_3):
        super(fizzbuzz_NN, self).__init__()
        self.linear_1 = nn.Linear(n_0, n_1)
        self.linear_2 = nn.Linear(n_1, n_2)
        self.linear_3 = nn.Linear(n_2, n_3)
    # Defining the forward propagation.
    def forward(self, x):
        x = self.linear_1(x)
        x = F.relu(x)
        x = self.linear_2(x)
        x = F.relu(x)
        x = self.linear_3(x)
        return x

# Defining the function for the evaluation.
def evaluation(devSet, devLabelSet, model, loss):
    # Evaluating the model.
    prediction = model(devSet)                                              # Doing the prediction.
    evalCost = loss(prediction, devLabelSet).item()                         # Getting the value of the loss.
    evalAccuracy = (torch.argmax(prediction, 1) == devLabelSet)             # Getting the value of the accuracy.
    evalAccuracy = (evalAccuracy.sum().float() / len(evalAccuracy)).item()  # Getting the value of the accuracy,
    return evalCost, evalAccuracy                                           # Return the cost and accuracy.

# Defining the function for the training.
def train(trainingSet, trainingLabelSet, devSet, devLabelSet, m = m, n_0 = n_0, n_1 = n_1, n_2 = n_2, n_3 = n_3, lr = learning_rate, epoch = epoch, bs = becth_size):
    # Preparing for training.
    model = fizzbuzz_NN(n_0, n_1, n_2, n_3).to(device)  # Creating the model.
    loss = nn.CrossEntropyLoss()                        # Setting the loss function.
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
            trainingData = trainingSet[start : (start + bs)]            # Getting each minibatch.
            trainingLabel = trainingLabelSet[start : (start + bs)]      # Getting the corresponding label.
            prediction = model(trainingData)                            # Doing the prediction.
            losses = loss(prediction, trainingLabel)                    # Getting the value of the loss.
            optimizer.zero_grad()                                       # Clearing the previous gradient.
            losses.backward()                                           # Applying the backward propagation.
            optimizer.step()                                            # Updating the parameters.
            costs.append(losses.item())                                 # Getting the value of cost function.
            accuracy = (torch.argmax(prediction, 1) == trainingLabel)   # Getting the value of the accuracy.
            accuracy = accuracy.sum().float() / len(accuracy)           # Getting the value of the accuracy.
            accuracies.append(accuracy.item())                          # Getting the value of accuracy.       
        evalCost, evalAccuracy = evaluation(devSet, devLabelSet, model.eval(), loss)    # Applying the evaluation.
        if len(evalAccuracies) == 0 or evalAccuracy > max(evalAccuracies):
            torch.save(model.train().state_dict(), './Simple_Neural_Network_with_Pytorch/Fizzbuzz_Game_Implementation/Fizzbuzz.pt')  # Saving the model.
            print("Model Saved")
        evalAccuracies.append(evalAccuracy)                             # Storing the accuracy of the evaluation for picking up the best model.
        print("The value of training loss for epoch " + str(epoch + 1) + " is: " + str(np.sum(costs) / len(costs)))
        print("The value of training accuracy for epoch " + str(epoch + 1) + " is: " + str(np.sum(accuracies) / len(accuracies)))
        print("The value of evaluation loss for epoch " + str(epoch + 1) + " is: " + str(evalCost))
        print("The value of evaluation accuracy for epoch " + str(epoch + 1) + " is: " + str(evalAccuracy))

# Training and testing the model.
if __name__ == "__main__":
    # Get the users' command.
    cmd = input("Please choose train a model or evaluate the model ('T' for train, 'E' for evaluate, 'Exit' for quit): ")
    # Handling with the command.
    while cmd != "Exit":
        # Handle the command.
        if cmd == 'T':
            # Getting the total number of the fizzbuzz game.
            m = input("Please input the upper bound of the fizzbuzz game (Integer Value, 'Exit' for quit)): ")
            while True:
                try:
                    m = int(m) + 1
                    n_0 = int(np.log2(m - 1))
                    # Initializating the training data.
                    trainData, devData, testData = generateData(getData(m - 1))                                                 # Generating all the data.
                    trainingSet = torch.tensor([Encoder(data, m - 1) for data in trainData], dtype = torch.float32).to(device)  # Getting the training data.
                    trainingLabelSet = labelEncoder(trainData).to(device)                                                       # Getting the training label.
                    devSet = torch.tensor([Encoder(data, m - 1) for data in devData], dtype = torch.float32).to(device)         # Getting the development data.
                    devLabelSet = labelEncoder(devData).to(device)                                                              # Getting the development label.
                    testSet = torch.tensor([Encoder(data, m - 1) for data in testData], dtype = torch.float32).to(device)       # Getting the testing data.
                    testLabelSet = labelEncoder(testData).to(device)                                                            # Getting the testing label.
                    # Doing the training.
                    train(trainingSet, trainingLabelSet, devSet, devLabelSet, m = m, n_0 = n_0)
                    # Evaluating the model.
                    model = fizzbuzz_NN(n_0, n_1, n_2, n_3)                                                                     # Creating the evaluating model.
                    model.load_state_dict(torch.load('./Simple_Neural_Network_with_Pytorch/Fizzbuzz_Game_Implementation/Fizzbuzz.pt')) # Loading the model.
                    evalCost, evalAccuracy = evaluation(testSet, testLabelSet, model.to(device).eval(), loss = nn.CrossEntropyLoss())
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
            # Evaluating the model.
            model = fizzbuzz_NN(n_0, n_1, n_2, n_3)                                                                     # Creating the evaluating model.
            try:
                model.load_state_dict(torch.load('./Simple_Neural_Network_with_Pytorch/Fizzbuzz_Game_Implementation/Fizzbuzz.pt')) # Loading the model.
                number = input("Please input a integer between " + str(0) + " and " + str(np.power(2, model.state_dict().get("linear_1.weight").shape[1])) + " ('Exit' for quit): ")
                while True:
                    try:
                        number = int(number)
                        if number > np.power(2, model.state_dict().get("linear_1.weight").shape[1]) or number < 0:
                            number = input("Invalid Input! Please input a integer between " + str(0) + " and " + str(np.power(2, model.state_dict().get("linear_1.weight").shape[1])) + " ('Exit' for quit): ")
                        else:
                            model.eval().to(device)
                            testData = torch.tensor(Encoder(number, np.power(2, model.state_dict().get("linear_1.weight").shape[1])), dtype = torch.float32).to(device)
                            testLabel = torch.argmax(model(testData))
                            print("The model predict the input is: " + labelDecoder(testLabel.item()))
                            print("The real label is: " + labelDecoder(fizzbuzz(number)))
                            if labelDecoder(testLabel) == labelDecoder(fizzbuzz(number)):
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