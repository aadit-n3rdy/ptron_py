import numpy as np
import scipy as sp

lrate = 0.5

def sigmoid(z: float):
    return 1 / (1 + np.e ** -z)


def sigmoid_p(z: float):
    return sigmoid(z) * (1 - sigmoid(z))


class layer:
    inpCount: int
    weights: np.ndarray
    z: np.ndarray
    a: np.ndarray
    da: np.ndarray
    def __init__(self,  prevLayerCount: int, curLayerCount: int):
        self.inpCount = prevLayerCount
        self.z = np.ndarray(curLayerCount)
        self.a = np.ndarray(curLayerCount)
        self.weights = np.random.rand(curLayerCount, prevLayerCount)
        self.da = np.zeros(curLayerCount)

    def calc(self, prevLayer: np.ndarray):
        if prevLayer.size != self.inpCount:
            print("Invalid input passed")
            exit(0)
        result = np.ndarray(self.nodes.size)
        z = self.weights.dot(prevLayer)
        '''for i in 0..self.z.size:
            self.z[i] = self.nodes[i].calc(prevLayer)
        self.z = result'''
        for i in 0..self.z.size:
            self.a[i] = sigmoid(self.z[i])
        return self.a

    def cost_func(self, ideal: np.ndarray):
        if ideal.size != self.nodes.size:
            exit(1)
        err = 0
        for i in ideal.size:
            err += abs(ideal[i] ** 2 - self.a[i] ** 2)

    def derivative_final(self, ideal: np.ndarray):
        if(ideal.size != self.a.size):
            exit(1)
        for i in 0..self.a.size:
            self.da[i] = ideal[i] - 1
        return self.da

    def derivative_hidden(self, next_z: np.ndarray, next_da: np.ndarray, next_weights: np.ndarray):
        for i in 0..self.da.size:
            self.da[i]=0
            for j in 0..next_da.size:
                self.da[i]+= next_weights[j][i] * sigmoid_p(next_z[j]) * next_da[j]
        return self.da

    def weight_adjust(self, previousLayer: np.ndarray):
        if(previousLayer.size != self.inpCount):
            exit(1)
        dw=0
        for i in 0..previousLayer.size:
            for j in 0..self.da.size:
                dw += self.da[j] * sigmoid_p(self.z[j] * previousLayer[i])
                self.weights[j][i] -= dw * lrate


def perceptron():
    shape = [1]
    learning_rate = 0.5
    def __init__(self, shp):
        shape = shp

    def learn(expected: np.ndarray):
        if(expected.size != shape[0]):
            print("Invalid size")
        else:


