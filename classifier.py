import numpy as np
import pandas as pd
import pickle as pkl

from losses import cross_entropy
from algorithms import sigmoid

class neural_net_classifier(object):

    def __init__(self, 
                 units_size, 
                 x, 
                 y, 
                 weights = False,
                 threshold = 0.5,
                 learning_rate = 0.05 
                 ):
        self.x = x
        self.y = y
        self.units_size = units_size
        self.no_of_samples = x.shape[1]
        self.weights = weights
        self.threshold = threshold
        self.parameters = {}
        self.learning_rate = learning_rate

    def __call__(self, epoch):
        """
        initializes weights if the weight are not 
        provided and calls forward

        Parameters:
        epoch = number of iterations for training
        """

        if not self.weights:
            self.initialize_parameters()
        
        self.train(epoch)
    
    def train(self, epoch):

        for i in range(epoch):
            print(f'{i}th iteration')
            activations = self.forward()
            print(cross_entropy(self.y, activations['aL']))
            gradients = self.backpropagate(activations)
            self.optimize(gradients, self.learning_rate)
    
    def initialize_parameters(self):
        """
        initializes weights based on the number of neurons in 
        each layer
        """

        for i in range(1, len(self.units_size)):
            self.parameters['w' + str(i)] = np.random.randn(self.units_size[i], self.units_size[i - 1]) * 0.01
            self.parameters['b' + str(i)] = np.zeros((self.units_size[i], 1))

            if i == len(self.units_size) - 1:
                self.parameters['wL'] =  self.parameters['w' + str(i)]
                self.parameters['bL'] = self.parameters['b' + str(i)]


            assert (self.parameters['w' + str(i)].shape[0] == self.units_size[i]) 
            print(f'shape of parameters: { self.parameters["w" + str(i)].shape}')
    
    
    def forward(self):
        """
        performs forward propogation

        Parameters:
        epoch = number of iterations to be performed for training

        Returns:
        dict of activation value of each layer 
        """

        activations = {}
        activations['a0'] = self.x

        for i in range(1, len(self.units_size)):
            activations['z' + str(i)] = np.dot(self.parameters['w' + str(i)], activations['a' + str(i - 1)]) + self.parameters['b' + str(i)] 

            if i == len(self.units_size) - 1:
                print(f'{i} sigmoid')
                assert(activations['z' + str(i)].shape[0] == 1)
                activations['a' + str(i)] = sigmoid(activations['z' + str(i)])
                activations['zL'] = activations['z' + str(i)]
                activations['aL'] = activations['a' + str(i)]
            else:
                assert(activations['z' + str(i)].shape[0] == self.parameters['w' + str(i)].shape[0])
                activations['a' + str(i)] = np.maximum(0, activations['z' + str(i)])

        return activations

    def backpropagate(self, activations):
        """
        calculates the gradient of the existing weights and bias

        Parameters:
        a1 = activations value in layer 1
        a2 = activations value in layer 2

        Returns:
        dict of Gradient value of weights and bias in all layer
        """

        gradients = {}
        dz_cache = {}

        for i in reversed(range(1, len(self.units_size))):

            if i == len(self.units_size) - 1:
                print(i, 'final layer')
                dz_cache['dz' + str(i)] = activations['a' + str(i)] - self.y
                dz_cache['dzL'] = dz_cache['dz' + str(i)]
            else:
                print(i)
                dz_cache['dz' + str(i)] = np.dot(self.parameters['w' + str(i + 1)].T, dz_cache['dz' + str(i + 1)])
                # print(dz_cache['dz' + str(i)][activations['z' + str(i)] <= 0])
                dz_cache['dz' + str(i)][activations['z' + str(i)] <= 0] = 0 

            gradients['dw' + str(i)] = np.dot(dz_cache['dz' + str(i)], activations['a' + str(i -1)].T) / self.no_of_samples
            gradients['db' + str(i)] = np.sum(dz_cache['dz' + str(i)], axis=1, keepdims=True) / self.no_of_samples

            if i == len(self.units_size) - 1:
                gradients['dwL'] = gradients['dw' + str(i)]
                gradients['dbL'] = gradients['db' + str(i)]
    
        return gradients

    def optimize(self, gradients, learning_rate):
        """
        updates the weights based on the derivatives

        Parameters:
        gradients: derivative of weights in each layer
        """

        for i in range(1, len(self.units_size)):
            self.parameters['w' + str(i)] = self.parameters['w' + str(i)] - (learning_rate * gradients['dw' + str(i)])
            self.parameters['b' + str(i)] = self.parameters['b' + str(i)] - (learning_rate * gradients['db' + str(i)])
        
            if i == len(self.units_size) - 1:
                self.parameters['wL'] =  self.parameters['w' + str(i)]
                self.parameters['bL'] = self.parameters['b' + str(i)]
    
    def save_parameters(self, parameters):
        """
        saves(serialize) the final parameters of the model, so that 
        we can reuse(deserialize) it for later prediciton 

        Parameters:
        weights: final parameters after the n epoch
        """

        pickle_obj = pkl.dumps(parameters)

        with open('model', 'w') as file:
            pkl.dump(pickle_obj, file)
