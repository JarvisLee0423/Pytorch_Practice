#============================================================================================#
#   Copyright:          JarvisLee                       
#   Date:               2020/11/24
#   Project Name:       AttentionGRUModel.py
#   Description:        Build the attention mechanism model to handle the English to Chinese
#                       machine translation problem.
#   Model Description:  Input               ->  Inputting English sentence
#                       Encoder             ->  Computing the context vector
#                       Attention           ->  Attention mechanism.
#                       Decoder             ->  Getting the translated Chinese sentence
#============================================================================================#

# Importing the necessary library.
import torch
import torch.nn as nn
import torch.nn.functional as F

# Creating the class for encoder.
class Encoder(nn.Module):
    # Creating the constructor.
    def __init__(self, vocabSize, hiddenSize):
        # Inheriting the super constructor.
        super(Encoder, self).__init__()
        # Creating the model.
        self.encoderEmbed = nn.Embedding(vocabSize, hiddenSize)
        self.encoderGRU = nn.GRU(hiddenSize, hiddenSize, batch_first = True, bidirectional = True)
        self.encoderDropout = nn.Dropout(0.2)
    # Creating the forward function.
    def forward(self, enData, enLength):
        # Applying the embedding layer. [batchSize, sequenceLength] -> [batchSize, sequenceLength, hiddenSize]
        embed = self.encoderEmbed(enData)
        # Applying the dropout. [batchSize, sequenceLength, hiddenSize] -> [batchSize, sequenceLength, hiddenSize]
        embed = self.encoderDropout(embed)
        # Unpadding the data. [batchSize, sequenceLength, hiddenSize] -> [batchSize, sentenceLength, hiddenSize]
        packedEmbed = nn.utils.rnn.pack_padded_sequence(embed, enLength, batch_first = True, enforce_sorted = False)
        # Applying the GRU. [batchSize, sentenceLength, hiddenSize] -> [batchSize, sentenceLength, 2 * hiddenSize]
        packedOutput, hidden = self.encoderGRU(packedEmbed)
        # Unpacking the data. [batchSize, sentenceLength, 2 * hiddenSize] -> [batchSize, sequenceLength, 2 * hiddenSize]
        output, _ = nn.utils.rnn.pad_packed_sequence(packedOutput, batch_first = True)
        # Returning the output and hidden. [1, batchSize, 2 * hiddenSize]
        return output, torch.cat([hidden[-1], hidden[-2]], 1).unsqueeze(0)

# Creating the class for attention.
class Attention(nn.Module):
    # Creating the constructor.
    def __init__(self, hiddenSize):
        # Inheriting the super constructor.
        super(Attention, self).__init__()
        # Creating the model.
        self.fc = nn.Linear(4 * hiddenSize, 2 * hiddenSize)
    # Creating the forward propagation.
    def forward(self, encoderOutput, decoderInput):
        # Setting the list to storing the context.
        context = []
        # Computing the context.
        for i in range(decoderInput.shape[1]):
            # Setting the list to storing the alpha.
            tempAlpha = []
            # Computing the alpha.
            for j in range(encoderOutput.shape[1]):
                # Concatenating the decoder's input with the encoder's output. [batchSize, 1, 2 * hiddenSize] -> [batchSize, 1, 4 * hiddenSize]
                x = torch.cat([decoderInput[:,i,:].unsqueeze(1), encoderOutput[:,j,:].unsqueeze(1)], 2)
                # Storing the partial alpha. [batchSize, 1, 4 * hiddenSize] -> encoderSequence * [batchSize, 1, 4 * hiddenSize]
                tempAlpha.append(x)
            # Getting the partial alpha. encoderSequence * [batchSize, 1, 4 * hiddenSize] -> [batchSize, encoderSequence, 4 * hiddenSize]
            alpha = torch.cat(tempAlpha, 1)
            # Computing the alpha. [batchSize * encoderSequence, 4 * hiddenSize] -> [batchSize * encoderSequence, 2 * hiddenSize] -> [batchSize, encoderSequence, 2 * hiddenSize]
            alpha = F.log_softmax(self.fc(alpha.reshape(-1, alpha.shape[2])), dim = 1).reshape(encoderOutput.shape[0], encoderOutput.shape[1], -1)
            # Computing the partial context. [batchSize, encoderSequence, 2 * hiddenSize] -> [batchSize, 1, 2 * hiddenSize]
            tempContext = torch.sum(alpha * encoderOutput, dim = 1).unsqueeze(1) 
            # Storing the partial context. [batchSize, 1, 2 * hiddenSize] -> decoderSequence * [batchSize, 1, 2 * hiddenSize]
            context.append(tempContext)
        # Getting the context. decoderSequence * [batchSize, 1, 2 * hiddenSize] -> [batchSize, decoderSequence, 2 * hiddenSize]
        context = torch.cat(context, 1)
        # Returning the context.
        return context

# Creating the class for decoder.
class Decoder(nn.Module):
    # Creating the constructor.
    def __init__(self, vocabSize, hiddenSize, attention):
        # Inheriting the super constructor.
        super(Decoder, self).__init__()
        # Creating the model.
        self.decoderEmbed = nn.Embedding(vocabSize, 2 * hiddenSize)
        self.decoderGRU = nn.GRU(2 * hiddenSize, 2 * hiddenSize, batch_first = True)
        self.attention = attention
        self.decoderLinear = nn.Linear(2 * hiddenSize, vocabSize)
        self.decoderDropout = nn.Dropout(0.2)
    # Creating the forward function.
    def forward(self, cnData, cnLength, output, hidden):
        # Applying the embedding layer. [batchSize, sequenceLength] -> [batchSize, sequenceLength, 2 * hiddenSize]
        embed = self.decoderEmbed(cnData)
        # Applying the dropout. [batchSize, sequenceLength, 2 * hiddenSize] -> [batchSize, sequenceLength, 2 * hiddenSize]
        embed = self.decoderDropout(embed)
        # Getting the context. [batchSize, sequenceLength, 2 * hiddenSize] -> [batchSize, sequenceLength, 2 * hiddenSize]
        context = self.attention(output, embed)
        # Unpadding the data. [batchSize, sequenceLength, hiddenSize] -> [batchSize, sentenceLength, hiddenSize]
        packedcontext = nn.utils.rnn.pack_padded_sequence(context, cnLength, batch_first = True, enforce_sorted = False)
        # Applying the GRU. [batchSize, sentenceLength, hiddenSize] -> [batchSize, sentenceLength, hiddenSize]
        packedOutput, hidden = self.decoderGRU(packedcontext, hidden)
        # Unpacking the data. [batchSize, sentenceLength, hiddenSize] -> [batchSize, sequenceLength, hiddenSize]
        output, _ = nn.utils.rnn.pad_packed_sequence(packedOutput, batch_first = True)
        # Reshaping the output. [batchSize, sequenceLength, hiddenSize] -> [batchSize * sequenceLength, hiddenSize]
        output = output.reshape(-1, output.shape[2])
        # Applying the linear. [batchSize * sequenceLength, hiddenSize] -> [batchSize * sequenceLength]
        output = self.decoderLinear(output)
        # Returning the output.
        return output, hidden

# Creating the Sequence to Sequence model.
class AttentionGRUModelNN(nn.Module):
    # Creating the constructor.
    def __init__(self, encoder, decoder):
        # Inheriting the super constructor.
        super(AttentionGRUModelNN, self).__init__()
        # Creating the model.
        self.encoder = encoder
        self.decoder = decoder
    # Creating the forward function.
    def forward(self, enData, enLength, cnData, cnLength):
        # Applying the encoder.
        output, hidden = self.encoder(enData, enLength)
        # Applying the decoder.
        output, _ = self.decoder(cnData, cnLength, output, hidden)
        # Returning the output.
        return output