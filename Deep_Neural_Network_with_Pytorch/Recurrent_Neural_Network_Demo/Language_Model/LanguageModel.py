#============================================================================================#
#   Copyright:          JarvisLee                       
#   Date:               2020/08/22
#   Project Name:       LanguageModel.py
#   Description:        Build the language model by using the pytorch.
#   Model Description:  Input               ->  Training text
#                       Embedding Layer     ->  Converting vocabulary size input into 
#                                               embedding size
#                       LSTM Layer          ->  Converting the embedding size input into
#                                               hidden size
#                       Linear Layer        ->  Converting the hidden size input into
#                                               vocabulary size
#                       Output              ->  The predicted word
#                       Train Target        ->  Predicting the next word according to the
#                                               current word
#============================================================================================#

# Importing the necessary library.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as tud
import torchtext
from torchtext.vocab import Vectors

# Fixing the computer device and random seed.
# Indicating whether the computer has the GPU or not.
if torch.cuda.is_available:
    # Fixing the random seed.
    torch.cuda.manual_seed(1)
    # Fixing the computer device.
    torch.cuda.set_device(0)
    # Setting the computer device.
    device = 'cuda'
else:
    # Fixing the random seed.
    torch.manual_seed(1)
    # Setting the computer device.
    device = 'cpu'

# Setting the hyperparameters.
# The value of the vocabulary size.
vocabularySize = 10000
# The value of the embedding size.
embeddingSize = 100
# The value of the hidden size.
hiddenSize = 100
# The value of the batch size.
batchSize = 128
# The value of the learning rate.
learningRate = 0.005
# The value of the epoch.
epoch = 10

# Creating the training data generator.
class dataGenerator():
    # Defining the data generating method.
    @staticmethod
    def generator(vocabularySize, batchSize):
        # Creating the text field which only contains the lower case letters.
        textField = torchtext.data.Field(lower = True)
        # Getting the training, development and testing data.
        trainData, devData, testData = torchtext.datasets.LanguageModelingDataset.splits(path = './Deep_Neural_Network_with_Pytorch/Recurrent_Neural_Network_Demo/Language_Model', train = 'train.txt', validation = 'dev.txt', test = 'test.txt', text_field = textField)
        # Creating the vocabulary.
        textField.build_vocab(trainData, max_size = vocabularySize)
        # Getting the training, development and testing sets.
        trainSet, devSet, testSet = torchtext.data.BPTTIterator.splits((trainData, devData, testData), batch_size = batchSize, device = device, bptt_len = 10, repeat = False, shuffle = True)
        # Returning the prepared data.
        return textField.vocab, trainSet, devSet, testSet

# Creating the model.
class LanguageModelNN(nn.Module):
    # Creating the constructor.
    def __init__(self, vocabularySize, embeddingSize, hiddenSize):
        # Inheritting the super constructor.
        super(LanguageModelNN, self).__init__()
        # Setting the embedding layer.
        self.embedding = nn.Embedding(vocabularySize, embeddingSize)
        # Setting the LSTM layer.
        self.lstm = nn.LSTM(embeddingSize, hiddenSize)
        # Setting the linear layer.
        self.linear = nn.Linear(hiddenSize, vocabularySize)
    # Defining the forward propagation.
    def forward(self, text, hidden):
        # Encoding the text by embedding layer. [timeStep, batchSize] -> [timeStep, batchSize, embeddingSize]
        text = self.embedding(text)
        # Feeding the data into the LSTM unit. [timeStep, batchSize, embeddingSize] -> [timeStep, batchSize, hiddenSize]
        text, hidden = self.lstm(text, hidden)
        # Reshaping the text to form the data whose second dimension is hiddenSize, in order to letting the linear layer to convert the size of all words into vocabularySize. [timeStep * batchSize, hiddenSize]
        text = text.reshape(-1, text.shape[2])
        # Feeding the data into the linear layer. [timeStep * batchSize, hiddenSize] -> [timeStep * batchSize, vocabularySize]
        text = self.linear(text)
        # Returning the predicted result of the forward propagation.
        return text, hidden
    # Initializing the input hidden.
    def initHidden(self, batchSize, hiddenSize, requireGrad = True):
        # Initializting the weight.
        weight = next(self.parameters())
        # Returning the initialized hidden. [1, batchSize, hiddenSize]: The first dimension represents num_layers * num_directions, num_layers represents how many lstm layers are applied, num_directions represents whether the lstm is bidirectional or not.
        return (weight.new_zeros((1, batchSize, hiddenSize), requires_grad = requireGrad), weight.new_zeros((1, batchSize, hiddenSize), requires_grad = requireGrad))
    # Spliting the historical hidden value from the current training.
    def splitHiddenHistory(self, hidden):
        # Checking whether the input is tensor or not.
        if isinstance(hidden, torch.Tensor):
            return hidden.detach()
        else:
            return tuple(self.splitHiddenHistory(h) for h in hidden)
    # Defining the training method.
    @staticmethod
    def trainer(model, optimizer, loss, trainSet, devSet, epoch, batchSize, hiddenSize):
        # Indicating whether the model is correct.
        assert type(model) != type(LanguageModelNN)
        # Setting the list to pick up the best model.
        evalAccs = []
        # Sending the model into the specific device.
        model = model.to(device)
        # Training the model.
        for epoch in range(epoch):
            # Initializing the training loss.
            trainLoss = []
            # Initializing the training accuracy.
            trainAcc = []
            # Initializing the hidden.
            hidden = model.initHidden(batchSize, hiddenSize)
            # Training the model.
            for i, trainData in enumerate(trainSet):
                # Remembering the historical hidden.
                hidden = model.splitHiddenHistory(hidden)
                # Feeding the data into the model.
                prediction, hidden = model(trainData.text, hidden)
                # Getting the value of the loss.
                cost = loss(prediction, trainData.target.view(-1))
                # Storing the cost.
                trainLoss.append(cost.item())
                # Clearing the previous gradient.
                optimizer.zero_grad()
                # Applying the backward propagation.
                cost.backward()
                # Updating the parameters.
                optimizer.step()
                # Computing the accuracy.
                accuracy = (torch.argmax(prediction, 1) == trainData.target.view(-1))
                accuracy = accuracy.sum().float() / len(accuracy)
                # Storing the accuracy.
                trainAcc.append(accuracy.item())
                # Printing the training information.
                if i % 100 == 0:
                    print("The iteration " + str(i) + ": Loss = " + str(cost.item()) + " || Acc = " + str(accuracy.item()))
            # Evaluating the model.
            evalLoss, evalAcc = LanguageModelNN.evaluator(model.eval(), loss, devSet, batchSize, hiddenSize)
            # Picking up the best model.
            if len(evalAccs) == 0 or evalAcc >= max(evalAccs):
                # Saving the model.
                torch.save(model.state_dict(), './Deep_Neural_Network_with_Pytorch/Recurrent_Neural_Network_Demo/Language_Model/LanguageModel.pt')
                print("Model Saved")
            # Converting the model mode.
            model.train()
            # Storing the evaluation accuracy.
            evalAccs.append(evalAcc)
            # Printing the training information.
            print("The epoch " + str(epoch + 1) + " training: Loss = " + str(np.sum(trainLoss) / len(trainLoss)) + " || Acc = " + str(np.sum(trainAcc) / len(trainAcc)))
            print("The epoch " + str(epoch + 1) + " evaluation: Loss = " + str(evalLoss) + " || Acc = " + str(evalAcc))
    # Defining the evaluation method.
    @staticmethod
    def evaluator(model, loss, devSet, bacthSize, hiddenSize):
        # Initializing the evaluation loss.
        evalLoss = []
        # Initializing the evaluation accuracy.
        evalAcc = []
        # Initializing the hidden.
        hidden = model.initHidden(batchSize, hiddenSize, requireGrad = False)
        # Evaluating the model.
        for i, evalData in enumerate(devSet):
            # Remembering the historical hidden.
            hidden = model.splitHiddenHistory(hidden)
            # Evaluating the model.
            prediction, hidden = model(evalData.text, hidden)
            # Computing the loss.
            cost = loss(prediction, evalData.target.view(-1))
            # Storing the loss.
            evalLoss.append(cost.item())
            # Computing the accuracy.
            accuracy = (torch.argmax(prediction, 1) == evalData.target.view(-1))
            accuracy = accuracy.sum().float() / len(accuracy)
            # Storing the accuracy.
            evalAcc.append(accuracy.item())
        # Returning the loss and accuracy.
        return np.sum(evalLoss) / len(evalLoss), np.sum(evalAcc) / len(evalAcc)

# Training the model.
if __name__ == "__main__":
    pass
    # # Generating the data.
    # vocab, trainSet, devSet, testSet = dataGenerator.generator(vocabularySize, batchSize)
    # # Creating the model.
    # model = LanguageModelNN(vocabularySize, embeddingSize, hiddenSize)
    # # Creating the loss function.
    # loss = nn.CrossEntropyLoss()
    # # Creating the optimizer.
    # optimizer = optim.Adam(model.parameters(), lr = learningRate)
    # cmd = input("Please input the command ('T' for training, 'E' for evaluation, 'Exit()' for quit): ")
    # # Handling the command.
    # while cmd != 'Exit()':
    #     if cmd == 'T':
    #         # Training the model.
    #         LanguageModelNN.trainer(model, optimizer, loss, trainSet, devSet, epoch, batchSize, hiddenSize)
    #         cmd = input("Please input the command ('T' for training, 'E' for evaluation, 'Exit()' for quit): ")
    #     elif cmd == 'E':
    #         try:
    #             # Loading the model.
    #             model.load_state_dict(torch.load('./Deep_Neural_Network_with_Pytorch/Recurrent_Neural_Network_Demo/Language_Model/LanguageModel.pt'))
    #             # Sending the model into the corresponding device.
    #             model = model.to(device)
    #             # Testing the model by perplexity.
    #             testLoss, _ = LanguageModelNN.evaluator(model.eval(), loss, testSet, batchSize, hiddenSize)
    #             # Printing the perplexity.
    #             print("The perplexity is: " + str(np.exp(testLoss)))
    #             # Getting the first word of the sentence.
    #             word = input("Please input the first word ('Exit()' for quit): ")
    #             while word != 'Exit()':
    #                 # Getting the first word.
    #                 wordIndex = vocab.stoi.get(str.lower(word), vocab.stoi.get('<unk>'))
    #                 # Initializing the sentence.
    #                 sentence = [vocab.itos[wordIndex]]
    #                 # Generating the input data.
    #                 data = torch.tensor([[wordIndex]]).to(device)
    #                 # Getting the prediction.
    #                 for i in range(10):
    #                     # Initializing the hidden.
    #                     hidden = model.initHidden(1, hiddenSize)
    #                     # Getting the prediction.
    #                     prediction, _ = model(data, hidden)
    #                     # Getting the index of the predicted word.
    #                     index = torch.argmax(prediction, 1)
    #                     # Storing the predicted word.
    #                     sentence.append(vocab.itos[index])
    #                     # Getting another data.
    #                     data = torch.tensor([[wordIndex]]).to(device)
    #                 # Printing the predicted sentence.
    #                 print("The predicted sentence is: " + " ".join(sentence))
    #                 # Getting another first word of the sentence.
    #                 word = input("Please input the first word ('Exit()' for quit): ")
    #         except:
    #             # Giving the hint.
    #             print("There are not any trained model, please train one first!!!")
    #             word = 'T'
    #         cmd = word
    #     else:
    #         cmd = input("Invalid Input! Please input the command ('T' for training, 'E' for evaluation, 'Exit()' for quit): ") 