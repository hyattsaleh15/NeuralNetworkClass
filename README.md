# NueralNetworkClass
Class created using tensorflow library intended to initialize and train a neural network

##Example to use it:

import NN_class

###Initialize the neural network class with the hyperparameters you prefer
model = NN([50,10,5],n=20,classes=1,learning_rate=1e-3)

On the example, the neural network has three layers of 50, 10 and 5 neurons. 
The data has 20 attributes with just 1 output class
The learning rate is se to 1e-3

###Train the model with your data

model.train(X_train,Y_train,X_test,Y_test,batch_size = 256,epochs = 100)
