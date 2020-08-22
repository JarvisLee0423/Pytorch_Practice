#============================================================================================#
#   Copyright:          JarvisLee                       
#   Date:               2020/08/20
#   Project Name:       NumpyNN.py
#   Description:        Using library numpy to build the 3-layers neural network.
#   Model Description:  Hypothesis 1    ->  n_1 nodes
#                       tanh            ->  Activation Function
#                       Hypothesis 2    ->  n_2 nodes
#                       tanh            ->  Activation Function
#                       Hypothesis 3    ->  n_3 nodes
#                       sigmoid         ->  Classifier
#============================================================================================#

# Importing the necessary library.
import numpy as np

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
learning_rate = 0.01
# The number of training epoches.  
epoch = 1000            

# Creating the model.
class NumpyModel3():
    # Defining the training method.
    @staticmethod
    def trainer(m = m, n_0 = n_0, n_1 = n_1, n_2 = n_2, n_3 = n_3, lr = learning_rate, epoch = epoch):
        # Preparing the training data.
        # The initialization of the training data.
        X = np.random.rand(n_0, m)
        # The initialization of the truth label.              
        Y = np.random.randint(0, 2, (n_3, m))
        # The weight of the first layer.   
        W_1 = np.random.rand(n_1, n_0)
        # The weight of the second layer.          
        W_2 = np.random.rand(n_2, n_1)
        # The weight of the third layer.          
        W_3 = np.random.rand(n_3, n_2)
        # The bias of the first layer.          
        b_1 = np.random.rand(n_1, 1)
        # The bias of the second layer.            
        b_2 = np.random.rand(n_2, 1)
        # The bias of the third layer.            
        b_3 = np.random.rand(n_3, 1)
        # Appling the normalization of the training data.
        X = (X - np.mean(X)) / np.std(X)

        # Applying the gradient descent.
        for each in range(epoch):
            # Forward propagation.
            # The hypothesis computation of the first layer.
            Z_1 = np.dot(W_1, X) + b_1
            # The activation computation of the first layer.                                                              
            A_1 = np.tanh(Z_1)
            # The hypothesis computation of the second layer.                                                                    
            Z_2 = np.dot(W_2, A_1) + b_2
            # The activation computation of the second layer.                                                           
            A_2 = np.tanh(Z_2)
            # The hypothesis computation of the third layer.                                                                      
            Z_3 = np.dot(W_3, A_2) + b_3
            # The activation computation of the classifier.                                                            
            A = 1 / (1 + np.exp(-Z_3))
            # The binary cross-entropy cost function.                                                              
            Cost = 1 / m * np.sum(-np.multiply(Y, np.log(A)) - np.multiply((1 - Y), np.log(1 - A)))
            # Printing the value of the cost function. 
            print("The loss of the epoch " + str(each + 1) + " is: " + str(Cost))                   

            # Backward propagation.
            # The derivative of Z_3.
            dZ_3 = A - Y
            # The derivative of W_3.                                                            
            dW_3 = 1 / m * np.dot(dZ_3, A_2.T)
            # The derivative of b_3.                                      
            dB_3 = 1 / m * np.sum(dZ_3, axis = 1, keepdims = True)
            # The derivative of Z_2.                  
            dZ_2 = np.multiply(np.dot(W_3.T, dZ_3), (1 - np.square(np.tanh(Z_2))))
            # The derivative of W_2.  
            dW_2 = 1 / m * np.dot(dZ_2, A_1.T)
            # The derivative of b_2.                                      
            dB_2 = 1 / m * np.sum(dZ_2, axis = 1, keepdims = True)
            # The derivative of Z_1.                  
            dZ_1 = np.multiply(np.dot(W_2.T, dZ_2), (1 - np.square(np.tanh(Z_1))))
            # The derivative of W_1.  
            dW_1 = 1 / m * np.dot(dZ_1, X.T)
            # The derivative of b_1.                                        
            dB_1 = 1 / m * np.sum(dZ_1, axis = 1, keepdims = True)                  

            # Updating the parameters.
            # The weight updating of W_1.
            W_1 = W_1 - lr * dW_1
            # The weight updating of W_2. 
            W_2 = W_2 - lr * dW_2
            # The weight updating of W_3.  
            W_3 = W_3 - lr * dW_3
            # The weight updating of b_1.   
            b_1 = b_1 - lr * dB_1
            # The weight updating of b_2.   
            b_2 = b_2 - lr * dB_2
            # The weight updating of b_3.  
            b_3 = b_3 - lr * dB_3
        # Getting the predicted lable.  
        A = A >= 0.5
        # Printing the value of the training accuracy.                                               
        print("The accuracy is: " + str(np.sum(A == Y) / len(A)))
        # Printing the value of the truth label.   
        print("The truth label is: " + str(Y))
        # Printing the value of the predicted label.                      
        print("The predicted label is: " + str(A))                  

# Training the model.
if __name__ == "__main__":
    NumpyModel3.trainer()