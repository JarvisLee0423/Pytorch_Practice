#============================================================================================#
#   Copyright:          JarvisLee                       
#   Date:               2020/10/02
#   Project Name:       Seq2SeqGRUModel.py
#   Description:        Build the sequence to sequence model to handle the English to Chinese
#                       machine translation problem.
#   Model Description:  Input               ->  Inputting English sentence
#                       Encoder             ->  Computing the context vector
#                       Decoder             ->  Getting the translated Chinese sentence
#============================================================================================#

# Importing the necessary library.
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import nltk
import jieba
from collections import Counter

# Fixing the device and random seed.
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
vocabSize = 15000
# Setting the hidden size.
hiddenSize = 100
# Setting the learning rate.
learningRate = 0.01
# Setting the batch size.
batchSize = 64
# Setting the number of epoch.
epoches = 100

# Creating the class for generating the necessary training component.
class trainComponentsGenerator():
    # Setting the method to read the training data.
    @staticmethod
    def dataReader(train = True):
        # Setting the data root.
        root = './Deep_Neural_Network_with_Pytorch/Recurrent_Neural_Network_Demo/Machine_Translation/'
        # Getting the corresponding data.
        if train:
            # Opening the file.
            trainFile = open(root + 'train.txt', 'r', encoding = 'utf-8')
            # Getting all the sentences.
            sentences = trainFile.readlines()
            # Setting the list to storing the sentences.
            enSentence = []
            cnSentence = []
            # Getting all the sentence.
            for each in sentences:
                # Saperating the English sentence and Chinese sentence.
                temp = each.split('\t')
                # Getting the English sentence and Chinese sentence.
                enSentence.append(temp[0])
                cnSentence.append(temp[1].split('\n')[0])
        else:
            # Opening the file.
            devFile = open(root + 'dev.txt', 'r', encoding = 'utf-8')
            # Getting all the sentences.
            sentences = devFile.readlines()
            # Setting the list to storing the sentences.
            enSentence = []
            cnSentence = []
            # Getting all the sentence.
            for each in sentences:
                # Saperating the English sentence and Chinese sentence.
                temp = each.split('\t')
                # Getting the English sentence and Chinese sentence.
                enSentence.append(temp[0])
                cnSentence.append(temp[1].split('\n')[0])
        # Returning the data.
        return enSentence, cnSentence
    # Setting the method to tokenize the sentence.
    @staticmethod
    def tokenizer(train = True):
        # Tokenizing the data.
        if train:
            # Getting the sentence.
            en, cn = trainComponentsGenerator.dataReader(train)
            # Setting the list to store the English data.
            enData = []
            # Spliting the English sentence.
            for sentence in en:
                # Getting the tokenized sentence.
                enData.append(['<bos>'] + nltk.word_tokenize(sentence) + ['<eos>'])
            # Setting the list to store the Chinese data.
            cnData = []
            # Spliting the Chinese sentence.
            for sentence in cn:
                # Getting the tokenized sentence.
                cnData.append(['<bos>'] + jieba.lcut(sentence) + ['<eos>'])
        else:
            # Getting the sentence.
            en, cn = trainComponentsGenerator.dataReader(train)
            # Setting the list to store the English data.
            enData = []
            # Spliting the English sentence.
            for sentence in en:
                # Getting the tokenized sentence.
                enData.append(['<bos>'] + nltk.word_tokenize(sentence) + ['<eos>'])
            # Setting the list to store the Chinese data.
            cnData = []
            # Spliting the Chinese sentence.
            for sentence in cn:
                # Getting the tokenized sentence.
                cnData.append(['<bos>'] + jieba.lcut(sentence) + ['<eos>'])
        # Returning the tokenized data.
        return enData, cnData
    # Setting the method to generating the vocabulary.
    @staticmethod
    def vocabGenerator(vocabSize):
        # Getting the tokenized data.
        en, cn = trainComponentsGenerator.tokenizer(True)
        # Setting the list to conut the English words' appearance frequency.
        enWordFreq = Counter()
        # Getting the English words' appearance frequency.
        for sentence in en:
            for word in sentence:
                # Checking whether the word is begin symbol or ending symbol.
                if word != '<bos>' and word != '<eos>':
                    enWordFreq[word] += 1
        # Getting the English vocabulary.
        enVocab = ['<unk>', '<pad>', '<bos>', '<eos>']
        for word in enWordFreq.most_common(vocabSize):
            enVocab.append(word[0])
        # Forming the word to index tool.
        enStoi = {}
        for each in enVocab:
            enStoi[each] = enVocab.index(each)
        # Setting the list to count the Chinese words' appearance frequency.
        cnWordFreq = Counter()
        # Getting the Chinese words' appearance frequency.
        for sentence in cn:
            for word in sentence:
                # Checking whether the word is begin symbol or ending symbol.
                if word != '<bos>' and '<eos>':
                    cnWordFreq[word] += 1
        # Getting the Chinese vocabulary.
        cnVocab = ['<unk>', '<pad>', '<bos>', '<eos>']
        for word in cnWordFreq.most_common(vocabSize):
            cnVocab.append(word[0])
        # Forming the word to index tool.
        cnStoi = {}
        for each in cnVocab:
            cnStoi[each] = cnVocab.index(each)
        # Returning the tools
        return enVocab, enStoi, cnVocab, cnStoi
    # Setting the method to seperate the minibatch.
    @staticmethod
    def minibacthGenerator(batchSize, shuffle = True, train = True):
        # Getting the data.
        en, _ = trainComponentsGenerator.tokenizer(train)
        # Getting the minibacth.
        minibatches = []
        # Getting the minibatches.
        for i in range(0, len(en), batchSize):
            minibatches.append(i)
        # Shuffle the minibatches.
        if shuffle:
            np.random.shuffle(minibatches)
        # Returning the minibatches
        return minibatches
    # Setting the method to forming the minibatch.
    @staticmethod
    def trainComponentsGenerator(batchSize, vocabSize, shuffle = True, train = True):
        # Getting the minibatches.
        minibatches = trainComponentsGenerator.minibacthGenerator(batchSize, shuffle, train)
        # Getting the tokenized data.
        en, cn = trainComponentsGenerator.tokenizer(train)
        # Getting the vocabulary tools.
        enItos, enStoi, cnItos, cnStoi = trainComponentsGenerator.vocabGenerator(vocabSize)
        # Setting the list to store the minibatch.
        enMinibatches, cnMinibatches, enLengths, cnLengths = [], [], [], []
        # Forming the minibatch.
        for i in minibatches:
            # Checking whehter the last minibatch is large enough.
            if (i + batchSize - 1) <= (len(en) - 1):
                # Getting the raw minibatch.
                enRawMinibatch = en[i : i + batchSize]
                cnRawMinibatch = cn[i : i + batchSize]
                enLength = [len(sent) for sent in enRawMinibatch]
                cnLength = [len(sent) for sent in cnRawMinibatch]
                # Getting the max length of English minibatch and Chinese minibatch.
                enMaxLength = max(enLength)
                cnMaxLength = max(cnLength)
                # Padding the English minibatch and digitalizing the minibatch.
                for j, sent in enumerate(enRawMinibatch):
                    # Checking whether the sentence is long enough.
                    while len(sent) < enMaxLength:
                        sent.append('<pad>')
                    # Digitalizing the minibatch.
                    sent = [enStoi.get(word, enStoi.get('<unk>')) for word in sent]
                    enRawMinibatch[j] = sent
                # Padding the Chinese minibatch and digitalizing the minibatch.
                for j, sent in enumerate(cnRawMinibatch):
                    # Checking whether the sentence is long enough.
                    while len(sent) < cnMaxLength:
                        sent.append('<pad>')
                    # Digitalizing the minibatch.
                    sent = [cnStoi.get(word, cnStoi.get('<unk>')) for word in sent]
                    cnRawMinibatch[j] = sent
            else:
                # Getting the raw minibatch.
                enRawMinibatch = en[i : (len(en) - 1)]
                cnRawMinibatch = cn[i : (len(cn) - 1)]
                enLength = [len(sent) for sent in enRawMinibatch]
                cnLength = [len(sent) for sent in cnRawMinibatch]
                # Getting the max length of English minibatch and Chinese minibatch.
                enMaxLength = max(enLength)
                cnMaxLength = max(cnLength)
                # Padding the English minibatch and digitalizing the minibatch.
                for j, sent in enumerate(enRawMinibatch):
                    # Checking whether the sentence is long enough.
                    while len(sent) < enMaxLength:
                        sent.append('<pad>')
                    # Digitalizing the minibatch.
                    sent = [enStoi.get(word, enStoi.get('<unk>')) for word in sent]
                    enRawMinibatch[j] = sent
                # Padding the Chinese minibatch and digitalizing the minibatch.
                for j, sent in enumerate(cnRawMinibatch):
                    # Checking whether the sentence is long enough.
                    while len(sent) < cnMaxLength:
                        sent.append('<pad>')
                    # Digitalizing the minibatch.
                    sent = [cnStoi.get(word, cnStoi.get('<unk>')) for word in sent]
                    cnRawMinibatch[j] = sent
            # Converting the minibatch into the tensor.
            enMinibatch = torch.LongTensor(np.array(enRawMinibatch).astype('int64'))
            cnMinibatch = torch.LongTensor(np.array(cnRawMinibatch).astype('int64'))
            # Storing the minibatch.
            enMinibatches.append(enMinibatch)
            cnMinibatches.append(cnMinibatch)
            enLengths.append(enLength)
            cnLengths.append(cnLength)
        # Returning the train components.
        return enItos, enStoi, cnItos, cnStoi, enMinibatches, cnMinibatches, enLengths, cnLengths

# Creating the class for encoder.
class Encoder(nn.Module):
    # Creating the constructor.
    def __init__(self, vocabSize, hiddenSize):
        # Inheriting the super constructor.
        super(Encoder, self).__init__()
        # Creating the model.
        self.encoderEmbed = nn.Embedding(vocabSize, hiddenSize)
        self.encoderGRU = nn.GRU(hiddenSize, hiddenSize, batch_first = True)
        self.encoderDropout = nn.Dropout(0.2)
    # Creating the forward function.
    def forward(self, enData, enLength):
        # Applying the embedding layer. [batchSize, sequenceLength] -> [batchSize, sequenceLength, hiddenSize]
        embed = self.encoderEmbed(enData)
        # Applying the dropout. [batchSize, sequenceLength, hiddenSize] -> [batchSize, sequenceLength, hiddenSize]
        embed = self.encoderDropout(embed)
        # Unpadding the data. [batchSize, sequenceLength, hiddenSize] -> [batchSize, sentenceLength, hiddenSize]
        packedEmbed = nn.utils.rnn.pack_padded_sequence(embed, enLength, batch_first = True, enforce_sorted = False)
        # Applying the GRU. [batchSize, sentenceLength, hiddenSize] -> [1, batchSize, hiddenSize]
        _, hidden = self.encoderGRU(packedEmbed)
        # Returning the hidden.
        return hidden

# Creating the class for decoder.
class Decoder(nn.Module):
    # Creating the constructor.
    def __init__(self, vocabSize, hiddenSize):
        # Inheriting the super constructor.
        super(Decoder, self).__init__()
        # Creating the model.
        self.decoderEmbed = nn.Embedding(vocabSize, hiddenSize)
        self.decoderGRU = nn.GRU(hiddenSize, hiddenSize, batch_first = True)
        self.decoderLinear = nn.Linear(hiddenSize, vocabSize)
        self.decoderDropout = nn.Dropout(0.2)
    # Creating the forward function.
    def forward(self, cnData, cnLength, hidden):
        # Applying the embedding layer. [batchSize, sequenceLength] -> [batchSize, sequenceLength, hiddenSize]
        embed = self.decoderEmbed(cnData)
        # Applying the dropout. [batchSize, sequenceLength, hiddenSize] -> [batchSize, sequenceLength, hiddenSize]
        embed = self.decoderDropout(embed)
        # Unpadding the data. [batchSize, sequenceLength, hiddenSize] -> [batchSize, sentenceLength, hiddenSize]
        packedEmbed = nn.utils.rnn.pack_padded_sequence(embed, cnLength, batch_first = True, enforce_sorted = False)
        # Applying the GRU. [batchSize, sentenceLength, hiddenSize] -> [batchSize, sentenceLength, hiddenSize]
        packedOutput, hidden = self.decoderGRU(packedEmbed, hidden)
        # Unpacking the data. [batchSize, sentenceLength, hiddenSize] -> [batchSize, sequenceLength, hiddenSize]
        output, _ = nn.utils.rnn.pad_packed_sequence(packedOutput, batch_first = True)
        # Reshaping the output. [batchSize, sequenceLength, hiddenSize] -> [batchSize * sequenceLength, hiddenSize]
        output = output.reshape(-1, output.shape[2])
        # Applying the linear. [batchSize * sequenceLength, hiddenSize] -> [batchSize * sequenceLength]
        output = self.decoderLinear(output)
        # Returning the output.
        return output, hidden

# Creating the Sequence to Sequence model.
class Seq2SeqGRUModelNN(nn.Module):
    # Creating the constructor.
    def __init__(self, encoder, decoder):
        # Inheriting the super constructor.
        super(Seq2SeqGRUModelNN, self).__init__()
        # Creating the model.
        self.encoder = encoder
        self.decoder = decoder
    # Creating the forward function.
    def forward(self, enData, enLength, cnData, cnLength):
        # Applying the encoder.
        hidden = self.encoder(enData, enLength)
        # Applying the decoder.
        output, _ = self.decoder(cnData, cnLength, hidden)
        # Returning the output.
        return output
    # Creating the training method.
    @staticmethod
    def trainer(enTrainSet, cnTrainSet, enTrainLen, cnTrainLen, enDevSet, cnDevSet, enDevLen, cnDevLen, enVocabSize, cnVocabSize, hiddenSize, batchSize, learningRate, epoches):
        # Creating the encoder.
        encoder = Encoder(enVocabSize, hiddenSize).to(device)
        # Creating the decoder.
        decoder = Decoder(cnVocabSize, hiddenSize).to(device)
        # Creating the sequence to sequence model.
        model = Seq2SeqGRUModelNN(encoder, decoder).to(device)
        # Creating the loss function.
        loss = nn.CrossEntropyLoss()
        # Creating the optimizer.
        optimizer = optim.Adam(model.parameters(), lr = learningRate, betas = [0.5, 0.999])
        # Setting the list to storing the evaluating accuracy.
        evalAcces = []
        # Training the model.
        for epoch in range(epoches):
            # Setting the list for storing the training loss and accuracy.
            trainLosses = []
            trainAcces = []
            for i, bacth in enumerate(enTrainSet):
                # Getting the training data.
                enBatch = bacth.to(device)
                enLength = enTrainLen[i]
                # Decoder do not need the last words for inputting.
                cnBatchIn = cnTrainSet[i][:, :-1].to(device)
                # Decoder do not need the first word for predicting.
                cnBatchOut = cnTrainSet[i][:, 1:].to(device)
                cnLength = [j - 1 for j in cnTrainLen[i]]
                # Training the model.
                # Getting the prediction.
                prediction = model(enBatch, enLength, cnBatchIn, cnLength)
                # Computing the loss.
                cost = loss(prediction, cnBatchOut.reshape(-1))
                # Storing the cost.
                trainLosses.append(cost.item())
                # Clearing the gradient.
                optimizer.zero_grad()
                # Applying the backward propagation.
                cost.backward()
                # Updating the parameters.
                optimizer.step()
                # Computing the accuracy.
                accuracy = (torch.argmax(prediction, 1) == cnBatchOut.reshape(-1))
                accuracy = accuracy.sum().float() / len(accuracy)
                # Storing the accuracy.
                trainAcces.append(accuracy.item())
            # Evalutaing the model.
            evalLoss, evalAcc = Seq2SeqGRUModelNN.evaluator(enDevSet, cnDevSet, enDevLen, cnDevLen, model.eval(), loss)
            # Printing the training information.
            print("The epoch [%d/%d]: Train Loss = [%.4f] ||  Train Acc = [%.4f] || Eval Loss = [%.4f] || Eval Acc = [%.4f]" % (epoch + 1, epoches, np.sum(trainLosses) / len(trainLosses), np.sum(trainAcces) / len(trainAcces), evalLoss, evalAcc))
            # Checking whether the model is the best model.
            if len(evalAcces) == 0 or evalAcc >= max(evalAcces):
                # Giving the hint for saving the model.
                print("Model Saved")
                # Saving the model.
                torch.save(encoder.train().state_dict(), './Deep_Neural_Network_with_Pytorch/Recurrent_Neural_Network_Demo/Machine_Translation/Seq2SeqEncoder.pt')
                torch.save(decoder.train().state_dict(), './Deep_Neural_Network_with_Pytorch/Recurrent_Neural_Network_Demo/Machine_Translation/Seq2SeqDecoder.pt')
            # Storing the evaluating accuracy.
            evalAcces.append(evalAcc)
            # Converting the model state.
            model = model.train()
    # Creating the function for evaluation.
    @staticmethod
    def evaluator(enDevSet, cnDevSet, enDevLen, cnDevLen, model, loss):
        # Setting the list for storing the evaluating loss and accuracy.
        evalLosses = []
        evalAcces = []
        # Evaluating the model.
        for i, batch in enumerate(enDevSet):
            # Getting the evaluating data.
            enBatch = batch.to(device)
            enLength = enDevLen[i]
            cnBatchIn = cnDevSet[i][:, :-1].to(device)
            cnBatchOut = cnDevSet[i][:, 1:].to(device)
            cnLength = [j - 1 for j in cnDevLen[i]]
            # Evaluting the model.
            prediction = model(enBatch, enLength, cnBatchIn, cnLength)
            # Computing the loss.
            cost = loss(prediction, cnBatchOut.reshape(-1))
            # Storing the loss.
            evalLosses.append(cost.item())
            # Computing the accuracy.
            accuracy = (torch.argmax(prediction, 1) == cnBatchOut.reshape(-1))
            accuracy = accuracy.sum().float() / len(accuracy)
            # Storing the accuracy.
            evalAcces.append(accuracy.item())
        # Returning the evaluating result.
        return np.sum(evalLosses) / len(evalLosses), np.sum(evalAcces) / len(evalAcces)

# Setting the main function.
if __name__ == "__main__":
    pass
    # # Getting the training data.
    # enItos, enStoi, cnItos, cnStoi, enTrainSet, cnTrainSet, enTrainLen, cnTrainLen = trainComponentsGenerator.trainComponentsGenerator(batchSize, vocabSize, shuffle = True, train = True)
    # # Getting the development data.
    # _, _, _, _, enDevSet, cnDevSet, enDevLen, cnDevLen = trainComponentsGenerator.trainComponentsGenerator(batchSize, vocabSize, shuffle = False, train = False)
    # # Checking the training data.
    # for i in range(len(enTrainSet)):
    #     for k in range(enTrainSet[i].shape[0]):
    #         print(" ".join([enItos[index] for index in enTrainSet[i][k]]))
    #         print(" ".join([cnItos[index] for index in cnTrainSet[i][k]]))
    #         # Getting the command.
    #         cmd = input("'Exit' for quit looking the training data: ")
    #         # Handling the command.
    #         if cmd == 'Exit':
    #             break
    #     if cmd == 'Exit':
    #         break
    # # Checking the development data.
    # for i in range(len(enDevSet)):
    #     for k in range(enDevSet[i].shape[0]):
    #         print(" ".join([enItos[index] for index in enDevSet[i][k]]))
    #         print(" ".join([cnItos[index] for index in cnDevSet[i][k]]))
    #         # Getting the command.
    #         cmd = input("'Exit' for quit looking the development data: ")
    #         # Handling the command.
    #         if cmd == 'Exit':
    #             break
    #     if cmd == 'Exit':
    #         break
    # # Getting the input command.
    # cmd = input("Please input the command ('T' for training, 'E' for evaluating, 'Exit' for quit): ")
    # # Handling the command.
    # while cmd != 'Exit':
    #     if cmd == 'T':
    #         # Training the model.
    #         Seq2SeqGRUModelNN.trainer(enTrainSet, cnTrainSet, enTrainLen, cnTrainLen, enDevSet, cnDevSet, enDevLen, cnDevLen, len(enItos), len(cnItos), hiddenSize, batchSize, learningRate, epoches)
    #         # Getting the command.
    #         cmd = input("Please input the command ('T' for training, 'E' for evaluating, 'Exit' for quit): ")
    #     elif cmd == 'E':
    #         # Checking whether there is the model or not.
    #         try:
    #             # Creating the model.
    #             encoder = Encoder(len(enItos), hiddenSize)
    #             decoder = Decoder(len(cnItos), hiddenSize)
    #             # Loading the model.
    #             encoder.load_state_dict(torch.load('./Deep_Neural_Network_with_Pytorch/Recurrent_Neural_Network_Demo/Machine_Translation/Seq2SeqEncoder.pt'))
    #             decoder.load_state_dict(torch.load('./Deep_Neural_Network_with_Pytorch/Recurrent_Neural_Network_Demo/Machine_Translation/Seq2SeqDecoder.pt'))
    #             # Making the model into evaluating state.
    #             encoder = encoder.to(device).eval()
    #             decoder = decoder.to(device).eval()
    #             # Getting the input English sentence.
    #             sentence = input("Please input the English sentence ('Exit' for quit): ")
    #             # Handling the sentence.
    #             while sentence != 'Exit':
    #                 # Setting the list to storing the translation.
    #                 translation = []
    #                 # Initializing the index.
    #                 index = 2
    #                 # Spliting the sentence.
    #                 enData = ['<bos>'] + nltk.word_tokenize(sentence) + ['<eos>']
    #                 # Getting the length of the input English data.
    #                 length = [len(enData)]
    #                 # Getting the evaluating data.
    #                 enData = torch.LongTensor(np.array([enStoi.get(word, enStoi.get('<unk>')) for word in enData]).astype('int64')).unsqueeze(0).to(device)
    #                 # Getting the context hidden.
    #                 hidden = encoder(enData, length)
    #                 # Getting the first trainslated word.
    #                 prediction = torch.LongTensor(np.array([cnStoi['<bos>']]).astype('int64')).unsqueeze(0).to(device)
    #                 # Getting the trainslation.
    #                 while cnItos[index] != '<eos>':
    #                     # Getting the prediction.
    #                     prediction, hidden = decoder(prediction, [1], hidden)
    #                     # Getting the index.
    #                     index = torch.argmax(prediction, 1)
    #                     # Getting the word.
    #                     if cnItos[index] != '<eos>':
    #                         translation.append(cnItos[index])
    #                     # Reseting the prediction.
    #                     prediction = torch.LongTensor(np.array([index]).astype('int64')).unsqueeze(0).to(device)
    #                 # Printing the tanslation.
    #                 print("The Chinese translation is: " + " ".join(translation))
    #                 # Getting the input English sentence.
    #                 sentence = input("Please input the English sentence ('Exit' for quit): ")
    #             # Quiting the system.
    #             cmd = sentence
    #         except:
    #             # Giving the hint.
    #             print("There is no model! Please training one first!")
    #             # Training the model.
    #             cmd = 'T'
    #     else:
    #         # Giving the hint.
    #         cmd = input("Invalid command! Please input the command ('T' for training, 'E' for evaluating, 'Exit' for quit): ")