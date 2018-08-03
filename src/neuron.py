class Neuron:
    def __init__(self, bias, multiply = None):
        # initialization
        self.bias = bias
        self.weights = []
        self.multiply = multiply

    def calculate_output(self, inputs):
        # get outputs by running inputs through weight function
        self.inputs = inputs
        self.output = self.squash(self.calculate_total_net_input())
        return self.output

    def calculate_total_net_input(self):
        total = 0
        # totaling the weighted inputs
        for i in range(len(self.inputs)):
            total += self.inputs[i] * self.weights[i]
        return total + self.bias

    # apply the logistic function to squash the output of the neuron
    # the result is sometimes referred to as 'net' [2] or 'net' [1]
    def squash(self, total_net_input):
        # self.multiply is false for the XOR problem but true for the multiplication problem
        if self.multiply == True:
            # return the total_net_input for multiplcation because you don't want to call an activation function
            return total_net_input
        global activation
        # activate sigmoid
        if activation == "sigmoid":
            return 1 / (1 + math.exp(-total_net_input))
        # actually leaky ReLu but I got lazy with naming when I was trying different calculations
        elif activation == "relu":
            if total_net_input < 0:
                # return corresponding leaky ReLu for multiplication
                return 0.01 * total_net_input
            else:
                return total_net_input

    # determine how much the neuron's total input has to change to move closer to the expected output:
    #
    # we have the partial derivative of the error with respect to the output (∂E/∂yⱼ) and
    # the derivative of the output with respect to the total net input (dyⱼ/dzⱼ) so we can calculate
    # the partial derivative of the error with respect to the total net input
    #
    # this value is also known as the delta (δ) [1]
    # δ = ∂E/∂zⱼ = ∂E/∂yⱼ * dyⱼ/dzⱼ
    def calculate_pd_error_wrt_total_net_input(self, target_output):
        return self.calculate_pd_error_wrt_output(target_output) * self.calculate_pd_total_net_input_wrt_input();

    # the error for each neuron is calculated by the Mean Square Error method:
    def calculate_error(self, target_output):
        return 0.5 * (target_output - self.output) ** 2

    # the partial derivate of the error with respect to actual output then is calculated by:
    # = 2 * 0.5 * (target output - actual output) ^ (2 - 1) * -1
    # = -(target output - actual output)
    #
    # note that the actual output of the output neuron is often written as yⱼ and target output as tⱼ so:
    # = ∂E/∂yⱼ = -(tⱼ - yⱼ)
    def calculate_pd_error_wrt_output(self, target_output):
        return -(target_output - self.output)

    # the total net input into the neuron is squashed using logistic function to calculate the neuron's output:
    # yⱼ = φ = 1 / (1 + e^(-zⱼ))
    # note that where ⱼ represents the output of the neurons in whatever layer we're looking at and ᵢ represents the layer below it
    #
    # the derivative (not partial derivative since there is only one variable) of the output then is:
    # dyⱼ/dzⱼ = yⱼ * (1 - yⱼ)
    def calculate_pd_total_net_input_wrt_input(self):
        # access global activation variable
        global activation
        total_net_input = self.calculate_total_net_input()
        # perform different tasks for sigmoid vs leaky ReLu
        if activation == "sigmoid":
            return self.output * (1 - self.output)
        elif activation == "relu":
            if total_net_input > 0:
                return 1
            else:
                # arbitrary value found online, any small value should have similar results
                return 0.01

    # the total net input is the weighted sum of all the inputs to the neuron and their respective weights:
    # = zⱼ = netⱼ = x₁w₁ + x₂w₂ ...
    #
    # The partial derivative of the total net input with respective to a given weight (with everything else held constant) then is:
    # = ∂zⱼ/∂wᵢ = some constant + 1 * xᵢw₁^(1-0) + some constant ... = xᵢ
    def calculate_pd_total_net_input_wrt_weight(self, index):
        return self.inputs[index]
