import numpy as np
import matplotlib.pyplot as plt
from numba import jit

def normalize(X):
    return X/max(X)

class neural_network:
    def init(self,shape):
        self.num_layers = len(shape)
        self.sizes = shape
        self.biases = [np.random.randn(y, 1) for y in shape[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(shape[:-1], shape[1:])]
    def e(self):
        print(self.sizes)
    @jit
    def sigmoid(self,z):
        s = 1 / (1 + np.exp(-z))
        return s
    def inv(self,z):
        s = np.log(z/(1-z))
        return s
    @jit
    def prime(self,z):
        s = self.sigmoid(z)*(1-self.sigmoid(z))
        return s
    def forwards_prop(self,X):
        activation = X
        activations = [X]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w,activation)+b[:,0]
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        Y = activations[-1]
        return Y
    def load_data(self,weight,bias):
        np_load_old = np.load
        np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
        self.weights = np.load(weight)
        self.biases = np.load(bias)
        print(np.shape(self.weights[0]))
    def test(self,X,Y):
        counter = 0
        activation = X
        activations = [X]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w,activation)+b[:,0]
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        if np.argmax(activations[-1]) == np.argmax(Y):
            counter = 1
        return counter
    def recog(self,X,Y):
        activation = X
        activations = [X]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w,activation)+b[:,0]
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        print(np.argmax(activations[-1]),np.argmax(Y))
        plt.imshow(np.reshape(X,[28,28]),cmap='hot')
        plt.show()
    def train(self,X,Y,eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        delta_nabla_b, delta_nabla_w = self.backprop(X,Y)
        nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        a = self.weights
        self.weights = [w-eta*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-eta*nb
                       for b, nb in zip(self.biases, nabla_b)]
#        print(np.sum(np.array(self.weights)-np.array(a))/7840,"delta")
    def backprop(self,X,Y):
        target = Y
        activation = X
        activations = [X]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w,activation)+b[:,0]
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        delta = np.zeros([self.sizes[-1],1])
        delta[:,0] = (activations[-1]-target) * self.prime(zs[-1])
#        print(delta)
        nabla_b[-1] = delta
        nabla_w[-1] = np.tensordot((activations[-1]-target) * self.prime(zs[-1]),activations[-2],axes=0)
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.prime(z)
            try:
                delta = np.dot(self.weights[-l+1].transpose(),delta[:,0]) * sp#128
            except IndexError:
                delta = np.dot(self.weights[-l+1].transpose(),delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.tensordot(delta, activations[-l-1],axes=0)
        print(round(0.5*np.sum(activations[-1]-target)**2,7))
        return (nabla_b, nabla_w)
    def save(self,weight,bias):
        np.save(weight,self.weights)
        np.save(bias,self.biases)
        print("saved!")