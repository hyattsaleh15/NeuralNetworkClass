import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from math import ceil

class HiddenLayer():
	'''
	Hidden layer class
	'''
    def __init__(self,n,units):
        self.W = tf.Variable(np.random.randn(n,units))
        self.b = tf.Variable(np.zeros(units))
        
    def forward(self,X):
        Z = tf.matmul(X,self.W) + self.b
        return tf.nn.relu(Z)
    
class OutputLayer():
	'''
	Output layer class
	'''
    def __init__(self,n,units):
        self.W = tf.Variable(np.random.randn(n,units))
        self.b = tf.Variable(np.zeros(units))
        
    def forward(self,X):
        return tf.matmul(X,self.W) + self.b
        
class NN():
	'''
	Neural network class
	'''
    def __init__(self,layer_units,n,classes,learning_rate):
        self.inputs = tf.placeholder(tf.float64,shape=(None,n))
        self.labels = tf.placeholder(tf.float64,shape=(None,1))
        
        self.layers = []
        for units in layer_units:
            self.layers.append(HiddenLayer(n,units))
            n = units
        self.layers.append(OutputLayer(n,classes))
        
        self.predictions = self.forward()
        self.cost = tf.losses.mean_squared_error(self.labels,self.predictions)
        
        self.train_op = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(self.cost)
        
        
    def forward(self):
        A = self.inputs
        
        for layer in self.layers:
            A = layer.forward(A)
            
        return A
    
    def train(self,X,Y,X_test,Y_test,batch_size = 256,epochs = 100):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        
        X,Y = shuffle(X,Y)
        Y = np.reshape(Y,(len(Y),1))
        Y_test = np.reshape(Y_test,(len(Y_test),1))
        batches = ceil(len(X) / batch_size)
        
        iteration = 0
        for e in range(epochs):
            for b in range(batches):
                X_train = X[(b*batch_size):(b+1)*batch_size,:]
                Y_train = Y[(b*batch_size):(b+1)*batch_size,:]

                _,predictions = sess.run((self.train_op,self.predictions),feed_dict={self.labels:Y_train,self.inputs:X_train})
                iteration += 1
                
                if iteration%100 == 0:
                    cost = sess.run(self.cost,feed_dict={self.labels:Y_train,self.inputs:X_train})
                    test_cost = sess.run(self.cost,feed_dict={self.labels:Y_test,self.inputs:X_test})
                    print(iteration,cost,test_cost,predictions)
