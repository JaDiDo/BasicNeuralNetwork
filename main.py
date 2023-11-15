import numpy as np
class Network:
    def __init__(self):
        #3 random weight for nodes
        self.weight =  np.random.random((3, 1))
    
    def train(self, given_input, given_output, trainingrange):
        for iteration in range(trainingrange):
            output = self.think(given_input)
            error = given_output - output
            adjustment = np.dot(given_input.T, error * self.sigmoid_derivative(output))
            self.weight += adjustment 
    def think(self, given_input):
        return self.sigmoid(np.dot(given_input, self.weight))
    def sigmoid_derivative(self, x):
        #derivative Sigmoid function
        return x * (1 - x)    
    def sigmoid(self, x):
        #sigmoid function
        return 1 / (1 + np.exp(-x))

training_inputs = np.array([[0,0,1],
                                [1,1,1],
                                [1,0,1],
                                [0,1,1]])
training_output = np.array([[1,0,0.7,0]]).T
nt = Network()
print(nt.weight)
nt.train(training_inputs, training_output, 10)
print(nt.weight)
print(nt.think([0.4,0.2,0]))