#============================================================================================#
#   Copyright:          JarvisLee
#   Date:               2020/08/20
#   Project Name:       torch_NN.py
#   Description:        Using library torch to build the 3-layers neural network.
#   Model Description:  Hypothesis 1    ->  n_1 nodes
#                       ReLu            ->  Activation Function
#                       Hypothesis 2    ->  n_2 nodes
#                       ReLu            ->  Activation Function
#                       Hypothesis 3    ->  n_3 nodes
#                       sigmoid         ->  Classifier 
#============================================================================================#

# Importing the necessary library.
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Setting the hyperparameters.
m = 100                 # The number of the training instances.
n_0 = 1000              # The number of features of the training data.
n_1 = 100               # The number of nodes of the first layer.
n_2 = 100               # The number of nodes of the second layer.
n_3 = 1                 # The number of nodes of the third layer.
learning_rate = 0.01    # The learning rate of the gradient descent.
epoch = 1000            # The number of training epoches.

# Indicate whether using the cuda or not, if yes, fix it.
if torch.cuda.is_available:
    device = 'cuda'
    torch.cuda.set_device(0)
else:
    device = 'cpu'

# Creating the class for the model.
class TorchModel_3(nn.Module):
    # Defining the constructor.
    def __init__(self, n_0, n_1, n_2, n_3):
        # Inheriting the torch model.
        super(TorchModel_3, self).__init__()
        # Defining the model.
        self.linear_1 = nn.Linear(n_0, n_1) # The hypothesis computation of the first layer.
        self.linear_2 = nn.Linear(n_1, n_2) # The hypothesis computation of the second layer.
        self.linear_3 = nn.Linear(n_2, n_3) # The hypothesis computation of the third layer.
    # Forward propagation.
    def forward(self, x):
        x = self.linear_1(x)    # The hypothesis computation of the first layer.
        x = F.relu(x)           # The activation computation of the first layer.
        x = self.linear_2(x)    # The hypothesis computation of the second layer.
        x = F.relu(x)           # The activation computation of the second layer.
        x = self.linear_3(x)    # The hypothesis computation of the third layer.
        return x                # Return the ouput value of the forward propagation.

# Defining the function to train the model.
def train(m = m, n_0 = n_0, n_1 = n_1, n_2 = n_2, n_3 = n_3, lr = learning_rate, epoch = epoch):
    # Initializing the training data.
    X = torch.randn((m, n_0))
    Y = torch.randint(0, 2, (m, n_3)).to(device)
    # Normalizing the training data.
    X = (X - torch.mean(X)) / torch.std(X)
    X = X.to(device)
    # Creating the model.
    model = TorchModel_3(n_0, n_1, n_2, n_3).to(device)
    # Getting the loss function.
    loss = nn.CrossEntropyLoss()
    # Getting the optimization.
    optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = 0.00005)
    # Optimizing the model.
    for epoch in range(epoch):
        prediction = model(X)   # Forward propagation.
        cost = loss(prediction, Y.squeeze()) # Computing the loss.
        print("The value of loss of epoch " + str(epoch + 1) + " is: " + str(cost.item()))  # Printing the value of the loss function.
        optimizer.zero_grad()   # Clear the previous gradient.
        cost.backward()         # Backward propagation.
        optimizer.step()        # Updating the parameters.
    accuracy = (torch.argmax(prediction, 1) == Y)                           # Computing the accuracy.
    accuracy = accuracy.sum().float() / len(accuracy)                       # Computing the accuracy
    print("The accuracy is: " + str(accuracy.item()))                       # Printing the accuracy.
    print("The predicted label is: " + str(torch.argmax(prediction, 1)))    # Printing the predicted label.
    print("The truth label is: " + str(Y))                                  # Printing the truth label.

# Training the model.
if __name__ == "__main__":
    train()