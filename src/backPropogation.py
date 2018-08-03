import random
import math
import numpy as np
import matplotlib.pyplot as plt
% matplotlib inline
from neuron import Neuron
from neuronLayer import NeuronLayer
from neuralNetwork import NeuralNetwork
# visualize progress bar
from tqdm import tqdm
# used global variable of activation to check when to use sigmoid vs leakyRelu
global activation

# plotting function
def plot_learning_curves(t,error, color='b'):
    plt.plot(t, error, color = color)

# run the file
def main():
    # training sets for the XOR problem only have 4 cases
    training_sets = [
        [[0, 0], [0]],
        [[0, 1], [1]],
        [[1, 0], [1]],
        [[1, 1], [0]]
    ]
    # call global variable to use sigmoid for this problem
    global activation
    activation = "sigmoid"
    nn = NeuralNetwork(len(training_sets[0][0]), 5, len(training_sets[0][1]), False)
    # can use large learning rate due to simple training
    nn.LEARNING_RATE = 1
    temp1, temp2 = [], []
    # bigger range = longer training = less error = better results
    for i in range(10000):
        training_inputs, training_outputs = random.choice(training_sets)
        nn.train(training_inputs, training_outputs)
        temp1.append(i)
        k = nn.calculate_total_error(training_sets)
        temp2.append(k)
        print(i, k)

    print()
    test1 = [0, 0]
    test2 = [0, 1]
    test3 = [1, 0]
    test4 = [1, 1]
    # verify performance
    print(nn.feed_forward(test1)[0])
    print(nn.feed_forward(test2)[0])
    print(nn.feed_forward(test3)[0])
    print(nn.feed_forward(test4)[0])
    print()

    plt.plot(temp1, temp2)
    nn.inspect()

    training_sets = []
    # much larger training set to account for many multiplications
    for i in range(1000):
        x = random.randint(-100, 100)
        y = random.randint(-100, 100)
        training_sets.append([[x / 100.0, y / 100.0], [(x * y) / 10000.0]])
    # call global variable to use relu for this problem
    global activation
    activation = "relu"
    nn = NeuralNetwork(len(training_sets[0][0]), 20, len(training_sets[0][1]), True)
    # smaller learning rate so error rate doesn't stagnate
    nn.LEARNING_RATE = .05
    temp1, temp2 = [], []
    # my computer's not fast enough to run a higher value so my error rate is high
    for i in range(10000):
        training_inputs, training_outputs = random.choice(training_sets)
        nn.train(training_inputs, training_outputs)
        temp1.append(i)
        k = nn.calculate_total_error(training_sets)
        temp2.append(k)
        print(i, k)

    plot_learning_curves(temp1, temp2)
    nn.inspect()

main()
