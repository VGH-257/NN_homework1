import numpy as np
from utils import get_activation, softmax, cross_entropy_loss, cross_entropy_grad

class MyModel:
    def __init__(self, input_size, hidden_sizes, output_size, activation='relu'):
        self.activation_name = activation
        self.act_fn, self.act_back = get_activation(activation)
        self.params = {}
        h1, h2 = hidden_sizes
        if activation == 'relu':
            self.params['W1'] = np.random.randn(input_size, h1) * np.sqrt(2. / input_size)
            self.params['W2'] = np.random.randn(h1, h2) * np.sqrt(2. / h1)
            self.params['W3'] = np.random.randn(h2, output_size) * np.sqrt(2. / h2)
        else:  # for 'tanh' or 'sigmoid'
            self.params['W1'] = np.random.randn(input_size, h1) * np.sqrt(1. / input_size)
            self.params['W2'] = np.random.randn(h1, h2) * np.sqrt(1. / h1)
            self.params['W3'] = np.random.randn(h2, output_size) * np.sqrt(1. / h2)
        self.params['b1'] = np.zeros(h1)
        self.params['b2'] = np.zeros(h2)
        self.params['b3'] = np.zeros(output_size)

    def forward(self, X):
        self.cache = {}
        self.cache['X'] = X

        z1 = X @ self.params['W1'] + self.params['b1']
        a1 = self.act_fn(z1)
        z2 = a1 @ self.params['W2'] + self.params['b2']
        a2 = self.act_fn(z2)
        scores = a2 @ self.params['W3'] + self.params['b3']

        self.cache.update({'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2, 'scores': scores})
        return scores

    def backward(self, probs, y_true, reg=0.0):
        grads = {}
        N = y_true.shape[0]
        dscore = cross_entropy_grad(probs, y_true)

        grads['W3'] = self.cache['a2'].T @ dscore / N + reg * self.params['W3']
        grads['b3'] = np.sum(dscore, axis=0) / N

        da2 = dscore @ self.params['W3'].T
        dz2 = self.act_back(da2, self.cache['z2'])
        grads['W2'] = self.cache['a1'].T @ dz2 / N + reg * self.params['W2']
        grads['b2'] = np.sum(dz2, axis=0) / N

        da1 = dz2 @ self.params['W2'].T
        dz1 = self.act_back(da1, self.cache['z1'])
        grads['W1'] = self.cache['X'].T @ dz1 / N + reg * self.params['W1']
        grads['b1'] = np.sum(dz1, axis=0) / N
        grads['X'] = dz1 @ self.params['W1'].T

        return grads
    
    def vis_backward(self, target_class=0):

        da2 = self.params['W3'][:,target_class:target_class+1].T# dscore @ self.params['W3'].T
        dz2 = da2

        da1 = dz2 @ self.params['W2'].T
        dz1 = da1
        grad_x = dz1 @ self.params['W1'].T

        return grad_x

    def save(self, path):
        np.savez(path, **self.params)

    def load(self, path):
        data = np.load(path)
        for k in self.params:
            self.params[k] = data[k]