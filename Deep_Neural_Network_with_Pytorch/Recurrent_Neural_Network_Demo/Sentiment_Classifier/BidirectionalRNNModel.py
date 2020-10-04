#============================================================================================#
#   Copyright:          JarvisLee                       
#   Date:               2020/08/24
#   Project Name:       BidirectionalRNNModel.py
#   Description:        Solve the binary sentiment classification problem (Good | Bad) 
#                       by bidirectional LSTM model.
#   Model Description:  Input               ->  Training sentence
#                       Embedding Layer     ->  Converting vocabulary size input into 
#                                               embedding size
#                       B-LSTM Layer        ->  Converting the embedding size input into
#                                               hidden size
#                       Linear Layer        ->  Computing the prediction
#============================================================================================#

# Importing the necessary library.
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
vocabularySize = 50000
# The value of the embedding size.
embeddingSize = 100
# The value of the hidden size.
hiddenSize = 100
# The number of classes.
classSize = 1
# The value of the learning rate.
learningRate = 0.01
# The value of the batch size.
batchSize = 64
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
class BidirectionalRNNModelNN(nn.Module):
    # Creating the constructor.
    def __init__(self, vocabularySize, embeddingSize, hiddenSize, classSize, padIndex):
        # Inheritting the super constructor.
        super(BidirectionalRNNModelNN, self).__init__()
        # Getting the padIndex which is used to pad all the sentences who are not long enough.
        self.padIndex = padIndex
        # Setting the embedding layer.
        self.embedding = nn.Embedding(vocabularySize, embeddingSize)
        # Set the dropout.
        self.dropout = nn.Dropout(p = 0.2)
        # Setting the bidirectional lstm.
        self.lstm = nn.LSTM(embeddingSize, hiddenSize, num_layers = 2, bidirectional = True, batch_first = True, dropout = 0.2)
        # Setting the first full-connected layer.
        self.linear = nn.Linear(2 * hiddenSize, classSize)
    # Defining the forward propagation.
    def forward(self, x, mask):
        # Applying the embedding layer. [batchSize, timeStep] -> [batchSize, timeStep, embeddingSize]
        x = self.embedding(x)
        # Applying the dropout.
        x = self.dropout(x)
        # Getting the sequence length.
        if type(mask) != type([]):
            length = mask.sum(1)
        else:
            length = mask
        # Unpacking the input. [batchSize, timeStep, embeddingSize] -> [batchSize, sentenceLength, embeddingSize]
        x = nn.utils.rnn.pack_padded_sequence(x, length, batch_first = True, enforce_sorted = False)
        # Applying the bidirectional lstm. [batchSize, sentenceLength, embeddingSize] -> [batchSize, sentenceLength, 2 * hiddenSize]
        x, _ = self.lstm(x)
        # Unpadding the input. [batchSize, sentenceLength, 2 * hiddenSize] -> [batchSize, timeStep, 2 * hiddenSize]
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first = True)
        # Getting the last timeStep's output. [batchSize, timeStep, 2 * hiddenSize] -> [batchSize, 2 * hiddenSize]
        x = x.mean(1).squeeze()
        # Applying the dropout.
        x = self.dropout(x)
        # Applying the linear layer. [batchSize, 2 * hiddenSize] -> [batchSize, 1]
        x = self.linear(x)
        # Returning the result. [batchSize, 1] -> [batchSize]
        return x.squeeze()
    # Defining the training method.
    @staticmethod
    def trainer(model, optimizer, loss, textField, trainSet, devSet, epoches):
        # Indicating whether the model is correct.
        assert type(model) != type(BidirectionalRNNModelNN)
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
                # Computing the padding mask. [batchSize, seqenceLength]
                text = trainData.text.permute(1, 0)
                # [batchSize, seqenceLength]
                mask = 1. - (text == textField.vocab.stoi[textField.pad_token]).float()
                # Feeding the data into the model.
                prediction = model(text, mask)
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
            evalLoss, evalAcc = BidirectionalRNNModelNN.evaluator(model.eval(), loss, textField, devSet)
            # Picking up the best model.
            if len(evalAccs) == 0 or evalAcc >= max(evalAccs):
                # Saving the model.
                torch.save(model.train().state_dict(), './Deep_Neural_Network_with_Pytorch/Recurrent_Neural_Network_Demo/Sentiment_Classifier/BidirectionalRNNModel.pt')
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
    def evaluator(model, loss, textField, devSet):
        # Initializing the evaluation loss.
        evalLoss = []
        # Initializing the evaluation accuracy.
        evalAcc = []
        # Evaluating the model.
        for i, devData in enumerate(devSet):
            # Computing the padding mask. [batchSize, seqenceLength]
            text = devData.text.permute(1, 0)
            # [batchSize, seqenceLength]
            mask = 1. - (text == textField.vocab.stoi[textField.pad_token]).float()
            # Evaluating the model.
            prediction = model(text, mask)
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
    # Generating the training data.
    textField, trainSet, devSet, testSet = dataGenerator.generator(vocabularySize, batchSize)
    # Creating the model, there are two extra parts in the vocabulary which are '<unk>' and '<pad>'.
    model = BidirectionalRNNModelNN(vocabularySize + 2, embeddingSize, hiddenSize, classSize, textField.vocab.stoi[textField.pad_token])
    # Customizing the initialized parameters of the embedding layer.
    # Getting the vocabulary as the vectors. 
    gloveVector = textField.vocab.vectors
    # Reinitializing the parameters of the embedding layer.
    model.embedding.weight.data.copy_(gloveVector)
    # Adding the '<unk>' and '<pad>' tokens into the parameters of the embedding layer.
    model.embedding.weight.data[textField.vocab.stoi[textField.pad_token]]
    model.embedding.weight.data[textField.vocab.stoi[textField.unk_token]]
    # Setting the optimizer.
    optimizer = optim.Adam(model.parameters(), lr = learningRate, weight_decay = 0.00005, betas = [0.5, 0.999])
    # Setting the loss function.
    loss = nn.BCEWithLogitsLoss()
    # Getting the command.
    cmd = input("Please input the command ('T' for training, 'E' for evaluating, 'Exit()' for quit): ")
    # Handling the command.
    while cmd != 'Exit()':
        # Handling the command.
        if cmd == 'T':
            # Training the model.
            BidirectionalRNNModelNN.trainer(model, optimizer, loss, textField, trainSet, devSet, epoches)
            cmd = input("Please input the command ('T' for training, 'E' for evaluating, 'Exit()' for quit): ")
        elif cmd == 'E':
            try:
                # Loading the model.
                model.load_state_dict(torch.load('./Deep_Neural_Network_with_Pytorch/Recurrent_Neural_Network_Demo/Sentiment_Classifier/BidirectionalRNNModel.pt'))
                # Sending the model into the corresponding computer device.
                model = model.to(device)
                # Testing the model.
                testLoss, testAcc = BidirectionalRNNModelNN.evaluator(model.eval(), loss, textField, testSet)
                # Printing the testing result.
                print("The testing: Loss = " + str(testLoss) + " || Acc = " + str(testAcc))
                # Getting the input sentence.
                sentence = input("Please input one sentiment sentence ('Exit() for quit'): ")
                while sentence != 'Exit()':
                    # Getting the words from the sentence.
                    words = [word for word in sentence.split()]
                    # Getting the index of the word.
                    wordsIndex = [textField.vocab.stoi[word] for word in words]
                    # Sending the words' index into the corresponding device.
                    wordsIndex = torch.LongTensor(wordsIndex).to(device).unsqueeze(0)
                    # Getting the prediction.
                    prediction = int(torch.sigmoid(model(wordsIndex, [len(words)])).item())
                    # Giving the predicted result.
                    if prediction == 0:
                        print("The sentence is negative sentiment! :(")
                    else:
                        print("The sentence is positive sentiment! :)")
                    # Getting the input sentence.
                    sentence = input("Please input one sentiment sentence ('Exit() for quit'): ")
            except:
                # Giving the hint.
                print("There are not any trained model, please train one first!!!")
                sentence = 'T'
            cmd = sentence
        else:
            cmd = input("Invalid Input! Please input the command ('T' for training, 'E' for evaluating, 'Exit() for quit'): ")