import numpy as np
import pickle as pkl

from utils import sigmoid_derivative, relu_derivative
from losses import cross_entropy
from algorithms import sigmoid

class neural_net_classifier(object):

    def __init__(self, 
                 units_size, 
                 x, 
                 y, 
                 weights = False,
                 threshold = 0.4,
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
        print("length",len(self.parameters) //2)
        self.train(epoch)
    
    def train(self, epoch):

        self.costs = []

        for i in range(epoch):
            print(f'{i}th iteration')
            aL, caches = self.forward(self.x)
            cost = cross_entropy(self.y, aL) 
            print(cost)
            gradients = self.backpropagate(aL, caches, self.y)
            self.optimize(gradients, self.learning_rate)

            if i % 100 == 0:
                self.costs.append(cost)
    
    def initialize_parameters(self):
        """
        initializes weights based on the number of neurons in 
        each layer
        """

        for i in range(1, len(self.units_size)):
            self.parameters['W' + str(i)] = np.random.randn(self.units_size[i], self.units_size[i - 1]) * 0.01
            self.parameters['b' + str(i)] = np.zeros((self.units_size[i], 1))
            assert (self.parameters['W' + str(i)].shape[0] == self.units_size[i]) 
            print(f'shape of parameters: { self.parameters["W" + str(i)].shape}')
    
    def forward(self, x):
        """
        performs forward propogation

        Parameters:
        epoch = number of iterations to be performed for training

        Returns:
        dict of activation value of each layer 
        """
    
        caches = []
        A_prev = x
        
        L = len(self.parameters) // 2  # number of layers in the neural network

        for i in range(1, L):
            W = self.parameters['W' + str(i)]
            b = self.parameters['b' + str(i)]

            z = np.dot(W, A_prev) + b

            linear_cache = (A_prev, W, b)
            activation_cache = z
            caches.append((linear_cache, activation_cache))

            a = np.maximum(0, z)
            A_prev = a

        W = self.parameters['W' + str(L)]
        b = self.parameters['b' + str(L)]
        zL = np.dot(W, A_prev) + b

        linear_cache = (A_prev, W, b)
        activation_cache = zL
        caches.append((linear_cache, activation_cache))

        aL = 1 / (1 + np.exp(-zL))

        return aL, caches

    def backpropagate(self, aL, caches, Y):
        """
        calculates the gradient of the existing weights and bias

        Parameters:
        aL = output of final layer
        caches = contains the values used to calculate the 'a' in each layer

        Returns:
        dict of Gradient value of weights and bias in all layer
        """

        L = len(caches)
        m = aL.shape[1]
        da = - (np.divide(Y, aL) - np.divide(1 - Y, 1 - aL))
        current_cache = caches[L - 1]
        linear_cache, activations_cache = current_cache
        grads = {}

        grads['dz' + str(L)] = sigmoid_derivative(da, activations_cache)
        grads['dW' + str(L)] = np.dot(grads['dz' + str(L)], linear_cache[0].T) / m
        grads['db' + str(L)] = np.sum(grads['dz' + str(L)], axis = 1, keepdims=True) / m
        grads['dA' + str(L - 1)] = np.dot(linear_cache[1].T, grads['dz' + str(L)])
        da = grads['dA' + str(L - 1)]
    
        for l in reversed(range(L - 1)):
            current_cache = caches[l]
            linear_cache, activations_cache = current_cache
            grads['dz' + str(l + 1)] = relu_derivative(da, activations_cache)
            grads['dW' + str(l + 1)] = np.dot(grads['dz' + str(l + 1)], linear_cache[0].T) / m
            grads['db' + str(l + 1)] = np.sum(grads['dz' + str(l + 1)], axis = 1, keepdims=True) / m
            grads['dA' + str(l)] = np.dot(linear_cache[1].T, grads['dz' + str(l + 1)])
            da = grads['dA' + str(l)]

        return grads

    def optimize(self, gradients, learning_rate):
        """
        updates the weights based on the derivatives

        Parameters:
        gradients: derivative of weights in each layer
        learning_rate: decides the pace at which parameters should optimize 
        """

        L = len(self.parameters) // 2 

        for l in range(L):
            self.parameters['W' + str(l + 1)] = self.parameters['W' + str(l + 1)] - (learning_rate * gradients['dW' + str(l + 1)])
            self.parameters['b' + str(l + 1)] = self.parameters['b' + str(l + 1)] - (learning_rate * gradients['db' + str(l + 1)])

    def predict(self, x, y):

        m = x.shape[1]
        probs, caches = self.forward(x)
    
        probs[probs < self.threshold] = 0
        probs[probs != 0] = 1

        print(f'accuracy = {np.sum(probs == y) / m }')

        return probs
        