from neuron import Neuron

class NeuronLayer:
    # pass multiply from neural network all the way down to neuron
    def __init__(self, num_neurons, bias, multiply = None):
        # Every neuron in a layer shares the same bias
        self.bias = bias if bias else random.random()
        self.neurons = []
        for i in range(num_neurons):
            # create neurons from the neuron class
            self.neurons.append(Neuron(self.bias, multiply))

    # print neurons and their information
    def inspect(self):
        print('Neurons:', len(self.neurons))
        for n in range(len(self.neurons)):
            print(' Neuron', n)
            for w in range(len(self.neurons[n].weights)):
                print('  Weight:', self.neurons[n].weights[w])
            print('  Bias:', self.bias)

    # calculate the output with the neuron class function
    def feed_forward(self, inputs):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.calculate_output(inputs))
        return outputs

    # get all of the outputs in the layer
    def get_outputs(self):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.output)
        return outputs
