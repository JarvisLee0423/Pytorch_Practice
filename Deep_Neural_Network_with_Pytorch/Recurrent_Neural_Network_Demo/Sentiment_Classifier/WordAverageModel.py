#============================================================================================#
#   Copyright:          JarvisLee                       
#   Date:               2020/08/22
#   Project Name:       WordAverageModel.py
#   Description:        Solve the binary sentiment classification problem (Good | Bad) 
#                       by word average model.
#   Model Description:  Input               ->  Training sentence
#                       Embedding Layer     ->  Converting vocabulary size input into 
#                                               embedding size
#                       Average Pooling     ->  Computing the encode for each sentence
#                       Linear Layer        ->  Converting the embedding size input into
#                                               hidden size
#                       ReLu                ->  Activation Function
#                       Linear Layer        ->  Computing the prediction.
#============================================================================================#

# Importing the necessary library.
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as tud
import torchtext
import torchtext.vocab as Vectors

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
    torch.cuda.manual_seed(1)
    # Setting the computer device.
    device = 'cpu'

# Setting the hyperparameters.
# The value of the vocabulary size.
vocabularySize = 10000
# The value of the embedding size.
embeddingSize = 100
# The value of the hidden size.
hiddenSize = 100
# The number of classes.
classSize = 1
# The value of the learning rate.
learningRate = 0.01
# The value of the batch size.
batchSize = 128
# The value of the epoch.
epoches = 10

# Creating the data generator.
class dataGenerator():
    # Defining the data generating method.
    @staticmethod
    def generator(vocabulaySize, batchSize):
        # Creating the text field.
        textField = torchtext.data.Field(tokenize = 'spacy', tokenizer_language = 'en_core_web_sm')
        # Creating the label field.
        labelField = torchtext.data.LabelField(dtype = torch.float32)
        # Getting the training and testing data.
        trainData, testData = torchtext.datasets.IMDB.splits(textField, labelField, root = './Datasets/IMDB/')
        # Spliting the training data into training data (70%) and development data (30%).
        trainData, devData = trainData.split(random_state = random.seed(1))
        # Creating the vocabulary.
        textField.build_vocab(trainData, max_size = vocabularySize, vectors = 'glove.6B.100d', unk_init = torch.Tensor.normal_, vectors_cache = './Datasets/IMDB/glove.6B.100d/')
        labelField.build_vocab(trainData)
        # Getting the train, dev and test sets.
        trainSet, devSet, testSet = torchtext.data.BucketIterator.splits((trainData, devData, testData), batch_size = batchSize, device = device)
        # Returning the training data.
        return textField, trainSet, devSet, testSet

# Creating the model.
class WordAverageModelNN(nn.Module):
    # Creating the constructor.
    def __init__(self, vocabularySize, embeddingSize, hiddenSize, classSize, padIndex):
        # Inheritting the super constructor.
        super(WordAverageModelNN, self).__init__()
        # Getting the padIndex which is used to pad all the sentences who are not long enough.
        self.padIndex = padIndex
        # Setting the embedding layer.
        self.embedding = nn.Embedding(vocabularySize, embeddingSize)
        # Setting the hidden layer.
        self.linear_1 = nn.Linear(embeddingSize, hiddenSize)
        # Setting the classifier.
        self.linear_2 = nn.Linear(hiddenSize, classSize)
    # Defining the forward propagation.
    def forward(self, x):
        # Applying the embedding layer. [timeStep, batchSize] -> [timeStep, batchSize, embeddingSize]
        x = self.embedding(x)
        # Reshaping the x which invers the first and second dimensions of the input x. [timeStep, batchSize, embeddingSize] -> [batchSize, timeStep, embeddingSize]
        x = x.permute(1, 0, 2)
        # Applying the average pooling to compute the encode of the input sentence. [batchSize, timeStep, embeddingSize] -> [batchSize, 1, embeddingSize]
        x = F.avg_pool2d(x, (x.shape[1], 1))
        # Squeezing the data. [batchSize, 1, embeddingSize] -> [batchSize, embeddingSize]
        x = x.squeeze(1)
        # Applying the first full-connected layer. [batchSize, embedding] -> [batchSize, hiddenSize]
        x = self.linear_1(x)
        # Applying the ReLu activation function.
        x = F.relu(x)
        # Applying the classifier. [batchSize, hiddenSize] -> [batchSize, classSize]
        x = self.linear_2(x)
        # Returning the prediction. [batchSize, classSize] -> [batchSize]
        return x.squeeze()
    # Defining the training method.
    @staticmethod
    def trainer(model, optimizer, loss, trainSet, devSet, epoches):
        # Indicating whether the model is correct.
        assert type(model) != type(WordAverageModelNN)
        # Initializing the evaluation accuracy.
        evalAccs = []
        # Sending the model into the corresponding device.
        model = model.to(device)
        # Training the model.
        for epoch in range(epoches):
            # Initializing the training loss.
            trainLoss = []
            # Initializing the training accuracy.
            trainAcc = []
            # Training the model.
            for i, trainData in enumerate(trainSet):
                # Feeding the data into the model.
                prediction = model(trainData.text)
                # Computing the loss.
                cost = loss(prediction, trainData.label)
                # Storing the loss.
                trainLoss.append(cost.item())
                # Clearing the previous gradient.
                optimizer.zero_grad()
                # Applying the backward propagation.
                cost.backward()
                # Updating the parameters.
                optimizer.step()
                # Computing the accuracy.
                accuracy = (torch.round(torch.sigmoid(prediction)) == trainData.label)
                accuracy = accuracy.sum().float() / len(accuracy)
                # Storing the accurcy.
                trainAcc.append(accuracy.item())
                # Printing the training information.
                if i % 100 == 0:
                    print("The iteration " + str(i) + ": Loss = " + str(cost.item()) + " || Acc = " + str(accuracy.item()))
            # Evaluating the model.
            evalLoss, evalAcc = WordAverageModelNN.evaluator(model.eval(), loss, devSet)
            # Picking up the best model.
            if len(evalAccs) == 0 or evalAcc >= max(evalAccs):
                # Saving the model.
                torch.save(model.train().state_dict(), './Deep_Neural_Network_with_Pytorch/Recurrent_Neural_Network_Demo/Sentiment_Classifier/WordAverageModel.pt')
                print("Model Saved")
            # Converting the model into train mode.
            model.train()
            # Storing the evaluation accuracy.
            evalAccs.append(evalAcc)
            # Printing the training information.
            print("The epoch " + str(epoch + 1) + " training: Loss = " + str(np.sum(trainLoss) / len(trainLoss)) + " || Acc = " + str(np.sum(trainAcc) / len(trainAcc)))
            print("The epoch " + str(epoch + 1) + " evaluating: Loss = " + str(evalLoss) + " || Acc = " + str(evalAcc))
    # Defining the evaluation method.
    @staticmethod
    def evaluator(model, loss, devSet):
        # Initializing the evaluation loss.
        evalLoss = []
        # Initializing the evaluation accuracy.
        evalAcc = []
        # Evaluating the model.
        for i, devData in enumerate(devSet):
            # Evaluating the model.
            prediction = model(devData.text)
            # Computing the loss.
            cost = loss(prediction, devData.label)
            # Storing the loss.
            evalLoss.append(cost.item())
            # Computing the accuracy.
            accuracy = (torch.round(torch.sigmoid(prediction)) == devData.label)
            accuracy = accuracy.sum().float() / len(accuracy)
            # Storing the accuracy.
            evalAcc.append(accuracy.item())
        # Returning the loss and accuracy.
        return np.sum(evalLoss) / len(evalLoss), np.sum(evalAcc) / len(evalAcc)

# Training the model.
if __name__ == "__main__":
    pass
    # # Generating the training data.
    # textField, trainSet, devSet, testSet = dataGenerator.generator(vocabularySize, batchSize)
    # # Creating the model, there are two extra parts in the vocabulary which are '<unk>' and '<pad>'.
    # model = WordAverageModelNN(vocabularySize + 2, embeddingSize, hiddenSize, classSize, textField.vocab.stoi[textField.pad_token])
    # # Customizing the initialized parameters of the embedding layer.
    # # Getting the vocabulary as the vectors. 
    # gloveVector = textField.vocab.vectors
    # # Reinitializing the parameters of the embedding layer.
    # model.embedding.weight.data.copy_(gloveVector)
    # # Adding the '<unk>' and '<pad>' tokens into the parameters of the embedding layer.
    # model.embedding.weight.data[textField.vocab.stoi[textField.pad_token]]
    # model.embedding.weight.data[textField.vocab.stoi[textField.unk_token]]
    # # Setting the optimizer.
    # optimizer = optim.Adam(model.parameters(), lr = learningRate)
    # # Setting the loss function.
    # loss = nn.BCEWithLogitsLoss()
    # # Getting the command.
    # cmd = input("Please input the command ('T' for training, 'E' for evaluating, 'Exit()' for quit): ")
    # # Handling the command.
    # while cmd != 'Exit()':
    #     # Handling the command.
    #     if cmd == 'T':
    #         # Training the model.
    #         WordAverageModelNN.trainer(model, optimizer, loss, trainSet, devSet, epoches)
    #         cmd = input("Please input the command ('T' for training, 'E' for evaluating, 'Exit()' for quit): ")
    #     elif cmd == 'E':
    #         try:
    #             # Loading the model.
    #             model.load_state_dict(torch.load('./Deep_Neural_Network_with_Pytorch/Recurrent_Neural_Network_Demo/Sentiment_Classifier/WordAverageModel.pt'))
    #             # Sending the model into the corresponding computer device.
    #             model = model.to(device)
    #             # Testing the model.
    #             testLoss, testAcc = WordAverageModelNN.evaluator(model.eval(), loss, testSet)
    #             # Printing the testing result.
    #             print("The testing: Loss = " + str(testLoss) + " || Acc = " + str(testAcc))
    #             # Getting the input sentence.
    #             sentence = input("Please input one sentiment sentence ('T' for training, 'Exit() for quit'): ")
    #             while sentence != 'Exit()':
    #                 # Getting the words from the sentence.
    #                 words = [word for word in sentence.split()]
    #                 # Getting the index of the word.
    #                 wordsIndex = [textField.vocab.stoi[word] for word in words]
    #                 # Sending the words' index into the corresponding device.
    #                 wordsIndex = torch.LongTensor(wordsIndex).to(device).unsqueeze(1)
    #                 # Getting the prediction.
    #                 prediction = int(torch.sigmoid(model(wordsIndex)).item())
    #                 # Giving the predicted result.
    #                 if prediction == 0:
    #                     print("The sentence is negative sentiment! :(")
    #                 else:
    #                     print("The sentence is positive sentiment! :)")
    #                 # Getting the input sentence.
    #                 sentence = input("Please input one sentiment sentence ('T' for training, 'Exit() for quit'): ")
    #         except:
    #             # Giving the hint.
    #             print("There are not any trained model, please train one first!!!")
    #             sentence = 'T'
    #         cmd = sentence
    #     else:
    #         cmd = input("Invalid Input! Please input the command ('T' for training, 'E' for evaluating, 'Exit() for quit'): ")