#============================================================================================#
#   Copyright:          JarvisLee                       
#   Date:               2020/09/23
#   Project Name:       Seq2SeqGRUModel.py
#   Description:        Achieve the machine translation by using sequence to sequence model
#                       with Gate Recurrent Unit.
#   Model Description:  Input               ->  Training sentence
#                       Embedding Layer     ->  Converting vocabulary index into embedding 
#                                               size
#                       Encoder             ->  Gate Recurrent Unit Encoder
#                       Decoder             ->  Gate Recurrent Unit Decoder
#============================================================================================#

# Importing the necessary library.
import numpy as np
import nltk
import jieba
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import Counter

# Fixing the random seed and device.
# Checking whether the computer has the corresponding device or not.
if torch.cuda.is_available():
    # Fixing the random seed.
    torch.cuda.manual_seed(1)
    # Fixing the device.
    torch.cuda.set_device(0)
    # Setting the device.
    device = 'cuda'
else:
    # Fixing the random seed.
    torch.manual_seed(1)
    # Setting the device.
    device = 'cpu'

# Setting the hyperparameters.
# Setting the vocabulary size.
vocabSize = 10000
# Setting the batch size.
batchSize = 64
# Setting the hidden size.
hiddenSize = 100
# Setting the learning rate.
learningRate = 0.01
# Setting the training epoches.
epoches = 20

# Setting the class for preprocessing the data.
class dataLoader():
    # Setting the data reader.
    @staticmethod
    def dataReader(dataType = 'train'):
        # Setting the data root.
        root = './Deep_Neural_Network_with_Pytorch/Recurrent_Neural_Network_Demo/Machine_Translation/' + dataType + '.txt'
        # Openning the data file.
        dataFile = open(root, 'r', encoding = 'utf8')
        # Getting the raw data.
        lines = dataFile.readlines()
        # Getting the English and Chinese sentence.
        enData = []
        cnData = []
        for each in lines:
            # Spliting the two sentence.
            tempData = each.split('\t')
            # Getting the English sentence.
            enData.append(tempData[0])
            # Getting the Chinese sentence.
            cnData.append(tempData[1].split('\n')[0])
        # Returning the data.
        return enData, cnData
    
    # Setting the tokenizer.
    @staticmethod
    def dataTokenizer(language = 'en', dataType = 'train'):
        # Getting the data.
        enData, cnData = dataLoader.dataReader(dataType)
        # Preparing the English data.
        if language == 'en':
            # Tokenizing the English data.
            enTokenizedData = [['<bos>'] + nltk.word_tokenize(s) + ['<eos>'] for s in enData]
            # Returning the tokenized data.
            return enTokenizedData
        # Preparing the Chinese data.
        if language == 'cn':
            # Tokenizing the Chinese data.
            cnTokenizedData = [['<bos>'] + jieba.lcut(s) + ['<eos>'] for s in cnData]
            # Returning the tokenized data.
            return cnTokenizedData
    
    # Creating the vocabulary.
    @staticmethod
    def vocabGenerator(language = 'en', dataType = 'train'):
        # Getting the tokenized data.
        data = dataLoader.dataTokenizer(language, dataType)
        # Counting the frequency.
        wordFreq = Counter()
        # Getting the word frequency.
        for sentence in data:
            for word in sentence:
                if word != '<bos>' and word != '<eos>':
                    wordFreq[word] += 1
        # Adding the '<unk>', '<pad>', '<bos>', '<eos>'.
        vocab = ['<unk>', '<pad>', '<bos>', '<eos>']
        # Getting the most common words.
        vocab.extend([word[0] for word in wordFreq.most_common(vocabSize)])
        # Creating the word to index.
        wordToIndex = {word:vocab.index(word) for word in vocab}
        # Creating the index to word.
        indexToWord = vocab
        # Returning the vocab.
        return vocab, indexToWord, wordToIndex, data
    
    # Creating the mini-batch.
    @staticmethod
    def batchCreator(data, lengths, tempMiniBatches):
        # Getting the final mini batch.
        miniBatches = [[] for i in range(len(tempMiniBatches))]
        dataLength = [[] for i in range(len(tempMiniBatches))]
        # Converting the sentence index into the sentence.
        for i, batch in enumerate(tempMiniBatches):
            # Getting the longest sentence's length.
            length = [lengths[index] for index in batch]
            maxLength = max(length)
            for sentIndex in batch:
                # Creatting the temporary sentence data.
                tempSentence = [1 for i in range(maxLength)]
                # Getting the sentence data.
                tempSentence[0:len(data[sentIndex])] = data[sentIndex]
                # Storing the sentence into the mini batch.
                miniBatches[i].append(tempSentence)
            # Converting the mini batch into tensor.
            miniBatches[i] = torch.tensor(np.array(miniBatches[i]), dtype = torch.int64)
            dataLength[i].extend(torch.tensor(np.array(length).astype('int64')))
        # Returning the mini batch.
        return miniBatches, dataLength

    # Generating the mini-batch.
    @staticmethod
    def batchGenerator(enData, enLengths, cnData, cnLengths, batchSize, shuffle = False):
        # Getting the batch index.
        batchIndex = np.arange(0, len(enLengths), batchSize)
        # Testing whether shuffle the batch index.
        if shuffle:
            np.random.shuffle(batchIndex)
        # Getting the sentence index for each batch.
        enTempMiniBatches = []
        cnTempMiniBatches = []
        for index in batchIndex:
            enTempMiniBatches.append(np.arange(index, min(index + batchSize, len(enLengths))))
            cnTempMiniBatches.append(np.arange(index, min(index + batchSize, len(cnLengths))))
        # Getting the English mini-batch.
        enMiniBatches, enDataLength = dataLoader.batchCreator(enData, enLengths, enTempMiniBatches)
        # Getting the Chinese mini-batch.
        cnMiniBatches, cnDataLength = dataLoader.batchCreator(cnData, cnLengths, cnTempMiniBatches)
        # Returning the data.
        return enMiniBatches, enDataLength, cnMiniBatches, cnDataLength

# Creating the encoder.
class Encoder(nn.Module):
    # Creating the constructor.
    def __init__(self, hiddenSize, enVocabSize, cnVocabSize):
        # Inheriting the super constructor.
        super(Encoder, self).__init__()
        # Setting the model for encoder.
        self.encoderEmbedding = nn.Embedding(enVocabSize, hiddenSize)
        self.encoderDropout = nn.Dropout(0.2)
        self.encoderGRU = nn.GRU(hiddenSize, hiddenSize, batch_first = True)

    # Setting the forward.
    def forward(self, enData, enLength):
        # Applying the encoder.
        # Applying the embedding layer. [batchSize, sequenceLength] -> [bacthSize, sequenceLength, hiddenSize]
        enEmbed = self.encoderEmbedding(enData)
        # Applying the dropout. [bacthSize, sequenceLength, hiddenSize] -> [bacthSize, sequenceLength, hiddenSize]
        enEmbed = self.encoderDropout(enEmbed)
        # Packing the en-Embedding. [bacthSize, sequenceLength, hiddenSize] -> [bacthSize, sequenceLength, hiddenSize]
        packedEnEmbed = nn.utils.rnn.pack_padded_sequence(enEmbed, enLength, batch_first = True, enforce_sorted = False)
        # Applying the Gate Recurrent Unit. [bacthSize, sequenceLength, hiddenSize] -> [1, batchSize, hiddenSize] Hints: The first dimension is the GRU's layers' # times its directions' #.
        _, hidden = self.encoderGRU(packedEnEmbed)
        # Returning the hidden.
        return hidden

# Creating the decoder.
class Decoder(nn.Module):
    # Creating the constructor.
    def __init__(self, hiddenSize, enVocabSize, cnVocabSize):
        # Inheriting the super constructor.
        super(Decoder, self).__init__()
        # Setting the model for decoder.
        self.decoderEmbedding = nn.Embedding(cnVocabSize, hiddenSize)
        self.decoderDropout = nn.Dropout(0.2)
        self.decoderGRU = nn.GRU(hiddenSize, hiddenSize, batch_first = True)
        self.decoderLinear = nn.Linear(hiddenSize, cnVocabSize)

    # Setting the forward.
    def forward(self, cnData, cnLength, hidden):
        # Applying the decoder.
        # Applying the embedding layer. [batchSize, sequenceLength] -> [bacthSize, sequenceLength, hiddenSize]
        cnEmbed = self.decoderEmbedding(cnData)
        # Applying the dropout. [bacthSize, sequenceLength, hiddenSize] -> [bacthSize, sequenceLength, hiddenSize]
        cnEmbed = self.decoderDropout(cnEmbed)
        # Packing the cn-Embeddig. [bacthSize, sequenceLength, hiddenSize] -> [bacthSize, sequenceLength, hiddenSize]
        packedCnEmbed = nn.utils.rnn.pack_padded_sequence(cnEmbed, cnLength, batch_first = True, enforce_sorted = False)
        # Applying the Gate Recurrent Unit. [bacthSize, sequenceLength, hiddenSize] -> [bacthSize, sequenceLength, hiddenSize]
        packedCnOutput, hidden = self.decoderGRU(packedCnEmbed, hidden)
        # Unpacking the output. [bacthSize, sequenceLength, hiddenSize] -> [bacthSize, sequenceLength, hiddenSize]
        output, _ = nn.utils.rnn.pad_packed_sequence(packedCnOutput, batch_first = True)
        # Reshaping the output. [bacthSize, sequenceLength, hiddenSize] -> [bacthSize * sequenceLength, hiddenSize]
        output = output.reshape(-1, output.shape[2])
        # Applying the linear layer. [bacthSize * sequenceLength, hiddenSize] -> [bacthSize * sequenceLength, cnVocabSize]
        output = self.decoderLinear(output)
        # Returning the output.
        return output, hidden 

# Creating the model.
class Seq2SeqGRUModelNN(nn.Module):
    # Creating the constructor.
    def __init__(self, encoder, decoder):
        # Inheriting the super constructor.
        super(Seq2SeqGRUModelNN, self).__init__()
        # Setting the model for encoder.
        self.encoder = encoder
        # Setting the model for decoder.
        self.decoder = decoder

    # Setting the forward.
    def forward(self, enData, enLength, cnData, cnLength):
        # Applying the encoder.
        hidden = self.encoder(enData, enLength)
        # Applying the decoder.
        output, _ = self.decoder(cnData, cnLength, hidden)
        # Returning the output.
        return output

    # Setting the training method.
    @staticmethod
    def trainer(hiddenSize, enVocabSize, cnVocabSize, enData, enLength, cnData, cnLength, enDevSet, enDevLen, cnDevSet, cnDevLen, learningRate, epoches):
        # Creating the model.
        encoder = Encoder(hiddenSize, enVocabSize, cnVocabSize)
        decoder = Decoder(hiddenSize, enVocabSize, cnVocabSize)
        model = Seq2SeqGRUModelNN(encoder, decoder)
        # Sending the model into the corresponding device.
        encoder = encoder.to(device)
        decoder = decoder.to(device)
        model = model.to(device)
        # Creating the loss function.
        loss = nn.CrossEntropyLoss()
        # Creating the optimizer.
        optimizerEncoder = optim.Adam(encoder.parameters(), lr = learningRate, weight_decay = 0.00005)
        optimizerDecoder = optim.Adam(decoder.parameters(), lr = learningRate, weight_decay = 0.00005)
        # Setting the list for evaluating accuracy of each epoch.
        evalAcces = []
        # Training the model.
        for epoch in range(epoches):
            # Setting the list to store the loss.
            trainLoss = []
            trainAcc = []
            # Training the model.
            for i, enBatch in enumerate(enData):
                # Getting the training data.
                enBatch = enBatch.to(device)
                cnBatch = cnData[i].to(device)
                # Feeding the data.
                prediction = model(enBatch, enLength[i], cnBatch, cnLength[i])
                # Computing the loss.
                cost = loss(prediction, cnBatch.view(-1))
                # Storing the cost.
                trainLoss.append(cost.item())
                # Computing the accuracy.
                accuracy = (torch.argmax(prediction, 1) == cnBatch.view(-1))
                accuracy = accuracy.sum().float() / len(accuracy)
                # Storing the accuracy.
                trainAcc.append(accuracy.item())
                # Clearing the gradient.
                optimizerEncoder.zero_grad()
                optimizerDecoder.zero_grad()
                # Applying the backword propagation.
                cost.backward()
                # Updating the parameters.
                optimizerEncoder.step()
                optimizerDecoder.step()
            # Evaluating the model.
            evalLoss, evalAcc = Seq2SeqGRUModelNN.evaluator(model.eval(), loss, enDevSet, enDevLen, cnDevSet, cnDevLen)
            # Selecting the best model.
            if len(evalAcces) == 0 or evalAcc >= max(evalAcces):
                # Saving the model.
                torch.save(encoder.train().state_dict(), './Deep_Neural_Network_with_Pytorch/Recurrent_Neural_Network_Demo/Machine_Translation/Seq2SeqEncoder.pt')
                torch.save(decoder.train().state_dict(), './Deep_Neural_Network_with_Pytorch/Recurrent_Neural_Network_Demo/Machine_Translation/Seq2SeqDecoder.pt')
                # Giving the hint.
                print("Model Saved")
            # Storing the evaluation accuracy.
            evalAcces.append(evalAcc)
            # Converting the model module.
            model = model.train()
            # Printing the training information.
            print("The epoch [%d/%d]: Train Loss: [%.3f] || Train Acc: [%.3f] || Eval Loss: [%.3f] || Eval Acc: [%.3f]" % (epoch + 1, epoches, np.sum(trainLoss) / len(trainLoss), np.sum(trainAcc) / len(trainAcc), evalLoss, evalAcc))

    # Setting the evaluation method.
    @staticmethod
    def evaluator(model, loss, enDevSet, enDevLen, cnDevSet, cnDevLen):
        # Setting the list to store the cost and accuracy.
        evalLoss = []
        evalAcc = []
        # Evaluation the model.
        for i, enBatch in enumerate(enDevSet):
            # Getting the evaluating data.
            enBatch = enBatch.to(device)
            cnBatch = cnDevSet[i].to(device)
            # Evaluating the model.
            prediction = model(enBatch, enDevLen[i], cnBatch, cnDevLen[i])
            # Computing the loss.
            cost = loss(prediction, cnBatch.view(-1))
            # Storing the loss.
            evalLoss.append(cost.item())
            # Computing the accuracy.
            accuracy = (torch.argmax(prediction, 1) == cnBatch.view(-1))
            accuracy = accuracy.sum().float() / len(accuracy)
            # Storing the accuracy.
            evalAcc.append(accuracy.item())
        # Returning the evaluation result.
        return np.sum(evalLoss) / len(evalLoss), np.sum(evalAcc) / len(evalAcc)

# Setting the main function.
if __name__ == "__main__":
    pass
    # # Getting the data.
    # cnVocab, cnItos, cnStoi, cnData = dataLoader.vocabGenerator('cn')
    # enVocab, enItos, enStoi, enData = dataLoader.vocabGenerator('en')
    # # Getting the training data.
    # cnTrainData = [[cnStoi.get(word, cnStoi.get('<unk>')) for word in s] for s in cnData]
    # enTrainData = [[enStoi.get(word, enStoi.get('<unk>')) for word in s] for s in enData]
    # # Getting the development data.
    # cnDevData = [[cnStoi.get(word, cnStoi.get('<unk>')) for word in s] for s in dataLoader.dataTokenizer('cn', 'dev')]
    # enDevData = [[enStoi.get(word, enStoi.get('<unk>')) for word in s] for s in dataLoader.dataTokenizer('en', 'dev')]
    # # Getting the length of each data.
    # cnTrainLen = [len(s) for s in cnTrainData]
    # enTrainLen = [len(s) for s in enTrainData]
    # cnDevLen = [len(s) for s in cnDevData]
    # enDevLen = [len(s) for s in enDevData]
    # # Getting the training set.
    # enTrainSet, enTrainLen, cnTrainSet, cnTrainLen = dataLoader.batchGenerator(enTrainData, enTrainLen, cnTrainData, cnTrainLen, batchSize, True)
    # # Getting the dev set.
    # enDevSet, enDevLen, cnDevSet, cnDevLen = dataLoader.batchGenerator(enDevData, enDevLen, cnDevData, cnDevLen, batchSize, False)
    # # Getting the input cmd
    # cmd = input("Please input the cmd ('T' for training, 'E' for Evaluating, 'Exit' for quit): ")
    # # Handling the cmd.
    # while cmd != 'Exit':
    #     if cmd == 'T':
    #         # Training the model.
    #         Seq2SeqGRUModelNN.trainer(hiddenSize, len(enVocab), len(cnVocab), enTrainSet, enTrainLen, cnTrainSet, cnTrainLen, enDevSet, enDevLen, cnDevSet, cnDevLen, learningRate, epoches)
    #         # Getting the cmd.
    #         cmd = input("Training Completed! Please input the cmd ('E' for Evaluating, 'Exit' for quit): ")
    #     elif cmd == 'E':
    #         try:
    #             # Creating the model.
    #             encoder = Encoder(hiddenSize, len(enVocab), len(cnVocab))
    #             decoder = Decoder(hiddenSize, len(enVocab), len(cnVocab))
    #             # Loading the model.
    #             encoder.load_state_dict(torch.load('./Deep_Neural_Network_with_Pytorch/Recurrent_Neural_Network_Demo/Machine_Translation/Seq2SeqEncoder.pt'))
    #             decoder.load_state_dict(torch.load('./Deep_Neural_Network_with_Pytorch/Recurrent_Neural_Network_Demo/Machine_Translation/Seq2SeqDecoder.pt'))
    #             # Sending the model to the corresponding device.
    #             encoder = encoder.eval().to(device)
    #             decoder = decoder.eval().to(device)
    #             # Getting the raw sentence.
    #             sentence = input("Please input an English sentence: ")
    #             # Tokenized the input sentence.
    #             tokenizedInput = ['<bos>'] + nltk.word_tokenize(sentence) + ['<eos>']
    #             # Converting the tokenized sentence into index.
    #             enTestData = torch.tensor([enStoi[word] for word in tokenizedInput], dtype = torch.int64).to(device).unsqueeze(0)
    #             # Getting the encodes.
    #             hidden = encoder(enTestData, torch.tensor([len(tokenizedInput)], dtype = torch.int64))
    #             # Getting the translation.
    #             # Setting the list for storing the translation.
    #             translatedSentence = []
    #             # Setting the counter to limiting the max length of the translated sentence.
    #             counter = 0
    #             # Setting the first word.
    #             prediction = torch.tensor([[cnStoi['<bos>']]], dtype = torch.int64).to(device)
    #             while cnItos[torch.argmax(prediction, 1).item()] != '<eos>':
    #                 # Increasing the length counter.
    #                 counter += 1
    #                 # Getting the first translated word.
    #                 prediction, hidden = decoder(torch.argmax(prediction, 1).unsqueeze(0), torch.tensor([1], dtype = torch.int64), hidden)
    #                 # Storing the translated word.
    #                 translatedSentence.append(cnItos[torch.argmax(prediction, 1).item()])
    #                 # Quiting translation.
    #                 if counter >= 10:
    #                     break
    #             # Printing the translated sentence.
    #             print("The Chinese sentence is: " + "".join(translatedSentence))
    #             # Getting the cmd.
    #             cmd = input("Please input the cmd ('T' for training, 'E' for Evaluating, 'Exit' for quit): ")
    #         except:
    #             # Giving the hints.
    #             print("There is no model! Please training one first!")
    #             # Setting the command.
    #             cmd = 'T'
    #     else:
    #         # Getting another command.
    #         cmd = input("Invalid cmd! Please input the cmd ('T' for training, 'E' for Evaluating, 'Exit' for quit): ")