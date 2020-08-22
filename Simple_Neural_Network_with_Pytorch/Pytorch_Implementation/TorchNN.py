#============================================================================================#
#   Copyright:          JarvisLee
#   Date:               2020/08/20
#   Project Name:       TorchNN.py
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
# The number of the training instances.
m = 100
# The number of features of the training data.            
n_0 = 1000
# The number of nodes of the first layer.            
n_1 = 100
# The number of nodes of the second layer.              
n_2 = 100
# The number of nodes of the third layer.               
n_3 = 1
# The learning rate of the gradient descent.                 
learning_rate = 1e-6
# The number of training epoches.    
epoch = 1000            

# Indicate whether using the cuda or not, if yes, fix it.
# Checking whether the computer has the GPU or not.
if torch.cuda.is_available:
    # Applying the GPU.     
    device = 'cuda'
    # Fixing the GPU.             
    torch.cuda.set_device(0)    
else:
    # Applying the CPU.
    device = 'cpu'              

# Creating the class for the model.
class TorchModel3(nn.Module):
    # Defining the constructor.
    def __init__(self, n_0, n_1, n_2, n_3):
        # Inheriting the torch model.
        super(TorchModel3, self).__init__()
        # The hypothesis computation of the first layer.    
        self.linear_1 = nn.Linear(n_0, n_1)
        # The hypothesis computation of the second layer.     
        self.linear_2 = nn.Linear(n_1, n_2)
        # The hypothesis computation of the third layer.     
        self.linear_3 = nn.Linear(n_2, n_3)     
    # Forward propagation.
    def forward(self, x):
        # The hypothesis computation of the first layer.
        x = self.linear_1(x)
        # The activation computation of the first layer.    
        x = F.relu(x)
        # The hypothesis computation of the second layer.           
        x = self.linear_2(x)
        # The activation computation of the second layer.    
        x = F.relu(x)
        # The hypothesis computation of the third layer.           
        x = self.linear_3(x)
        # Getting the classification.
        x = F.sigmoid(x)
        # Return the ouput value of the forward propagation.   
        return x                
    # Defining the training method.
    @staticmethod
    def trainer(m = m, n_0 = n_0, n_1 = n_1, n_2 = n_2, n_3 = n_3, lr = learning_rate, epoch = epoch):
        # Preparing the training data.
        # Initializing the training data.
        X = torch.randn((m, n_0))
        # Initializing the training data.                                                              
        Y = torch.randint(0, 2, (m, n_3), dtype = torch.float32).to(device)
        # Normalizing the training data.                                            
        X = (X - torch.mean(X)) / torch.std(X)
        # Sending the data into GPU.                                                  
        X = X.to(device)
        # Creating the model.                                                                        
        model = TorchModel3(n_0, n_1, n_2, n_3).to(device)
        # Getting the loss function.                                      
        loss = nn.MSELoss(reduction = "sum")
        # Getting the optimization.                                                            
        optimizer = optim.Adam(model.parameters(), lr = lr)
        # Optimizing the model.             
        for epoch in range(epoch):
            # Forward propagation.                                                              
            prediction = model(X)
            # Computing the loss.                                                               
            cost = loss(prediction, Y.squeeze())
            # Printing the value of the loss function.                                                
            print("The value of loss of epoch " + str(epoch + 1) + " is: " + str(cost.item()))
            # Clear the previous gradient.  
            optimizer.zero_grad()
            # Backward propagation.                                                             
            cost.backward()
            # Updating the parameters.                                                                     
            optimizer.step()
        # Computing the accuracy.                                                                    
        accuracy = (torch.argmax(prediction, 1) == Y)
        # Computing the accuracy                                           
        accuracy = accuracy.sum().float() / len(accuracy)
        # Printing the accuracy.                                       
        print("The accuracy is: " + str(accuracy.item()))
        # Printing the predicted label.                                       
        print("The predicted label is: " + str(torch.argmax(prediction, 1)))
        # Printing the truth label.                   
        print("The truth label is: " + str(Y))                                                  

# Training the model.
if __name__ == "__main__":
    TorchModel3.trainer()