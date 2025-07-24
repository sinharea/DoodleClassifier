from math import exp
import numpy as np
import pickle


# -- Activation function ---

import math

def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    else:
        z = math.exp(x)
        return z / (1 + z)


# Convert python function to numpy vector function
sigmoid_vectorized = np.vectorize(sigmoid)


def dsigmoid(x):
    # derivative of sigmoid function
    # return sigmoid(x) * (1 - sigmoid(x))
    # outputs hav already been passed through sigmoid function so we can just -> return x * (1 - x)

    return x * (1 - x)


# Convert python function to numpy vector function
dsigmoid_vectorized = np.vectorize(dsigmoid)


# --------------------------


class NeuralNetwork:
    def __init__(self, *number_of_nodes_in_each_layer):
        # Turning number_of_nodes_in_each_layer to a python List
        self.number_of_nodes_in_each_layer = list(number_of_nodes_in_each_layer)

        self.number_of_layers = len(number_of_nodes_in_each_layer)

        # note: input layer and output layer are not hidden layers
        self.number_of_hidden_layers = self.number_of_layers - 2

        # Creating empty lists that are going to filled when we do feedforward or Backpropagation
        self.layer = [[] for _ in range(self.number_of_layers)]
        self.layer_error = [[] for _ in range(self.number_of_layers)]
        self.layer_gradient = [[] for _ in range(self.number_of_layers)]
        self.weight_delta = [[] for _ in range(self.number_of_layers)]

        self.weight = []
        self.bias = []

        for i in range(self.number_of_layers - 1):
            # initializing i-th layer to (i + 1)-th layer weights (with random numbers between -1 and 1)
            self.weight.append(
                np.random.uniform(-1, 1,
                                  (self.number_of_nodes_in_each_layer[i + 1], self.number_of_nodes_in_each_layer[i])))

            # initializing i-th layer to (i + 1)-th layer biases (with random numbers between -1 and 1)
            self.bias.append(np.random.uniform(-1, 1, (self.number_of_nodes_in_each_layer[i + 1], 1)))

        # learning rate is a hyper-parameter so you can set it to any number you want
        # you have to find the learning rate that works for you
        self.learning_rate = 0.1

    def predict(self, inputs_list):
        # FeedForward

        # Turning input array (type: Python List) in to a numpy array
        # self.layer[0] refers to first layer which is input layer
        self.layer[0] = np.array(inputs_list)
        self.layer[0] = self.layer[0].reshape((self.number_of_nodes_in_each_layer[0], 1))

        # Calculating the value of each node in the i-th layer
        for i in range(1, self.number_of_layers):
            self.layer[i] = np.dot(self.weight[i - 1], self.layer[i - 1])
            self.layer[i] = np.add(self.layer[i], self.bias[i - 1])
            self.layer[i] = sigmoid_vectorized(self.layer[i])

        # self.layer[-1] refers to last layer which is output layer
        # output = self.layer[-1].tolist()[0]
        output = self.layer[-1].reshape(self.number_of_nodes_in_each_layer[-1],).tolist()
        return output

    def train(self, inputs_list, targets_list):
        # FeedForward

        # Turning input array (type: Python List) in to a numpy array
        # self.layer[0] refers to first layer which is input layer
        self.layer[0] = np.array(inputs_list)
        self.layer[0] = self.layer[0].reshape((self.number_of_nodes_in_each_layer[0], 1))

        # Calculating the value of each node in the i-th layer
        for i in range(1, self.number_of_layers):
            self.layer[i] = np.dot(self.weight[i - 1], self.layer[i - 1])
            self.layer[i] = np.add(self.layer[i], self.bias[i - 1])
            self.layer[i] = sigmoid_vectorized(self.layer[i])

        # Backpropagation

        # Turning targets array (type: Python List) in to a numpy array
        targets = np.array(targets_list)
        targets = targets.reshape((self.number_of_nodes_in_each_layer[-1], 1))

        # adjusting weights and biases
        for i in range(self.number_of_layers - 1, 0, -1):

            if i == (self.number_of_layers - 1):
                # we only have targets for output layer so we only can do this when i == output layer
                self.layer_error[i] = np.subtract(targets, self.layer[i])

            else:
                # calculate layers errors
                # Error of the i-th layer is basically the dot product of (i + 1)-the layer errors and transposed weights (form i-th layer to (i + 1)-th layer)
                # we calculate the dot product because, effect of a layer node on the error of each node in ->
                # the layer after them is determined by amount of weight between two nodes (hopefully that makes sense)
                # .T transposes the array
                self.layer_error[i] = np.dot(self.weight[i].T, self.layer_error[i + 1])

            # Calculate i-th layer gradients
            self.layer_gradient[i] = dsigmoid_vectorized(self.layer[i])
            self.layer_gradient[i] = np.multiply(self.layer_gradient[i], self.layer_error[i])
            self.layer_gradient[i] = np.multiply(self.layer_gradient[i], self.learning_rate)

            # Calculate weights delta (from (i - 1)i-th layer to i-th layer)
            # Weights delta is the amount of the change that we are going to apply to the weights
            # .T transposes the array
            self.weight_delta[i - 1] = np.dot(self.layer_gradient[i], self.layer[i - 1].T)

            #
            # Adding weight deltas and adjusting biases
            self.weight[i - 1] = np.add(self.weight[i - 1], self.weight_delta[i - 1])
            self.bias[i - 1] = np.add(self.bias[i - 1], self.layer_gradient[i])


    def save(self, filename="model.pkl"):
        try:
            with open(filename, "wb") as f:
                pickle.dump({
                    "weights": self.weight,
                    "biases": self.bias,
                    "architecture": self.number_of_nodes_in_each_layer,
                    "learning_rate": self.learning_rate  # Save learning rate for consistency
                }, f)
            print(f"Model successfully saved to {filename}")
        except Exception as e:
            raise Exception(f"Error saving model to {filename}: {str(e)}")

    @staticmethod
    def load(filename="model.pkl"):
        try:
            with open(filename, "rb") as f:
                data = pickle.load(f)
                
                # Validate loaded data
                if not all(key in data for key in ["weights", "biases", "architecture"]):
                    raise ValueError("Invalid model file: missing required keys")
                
                arch = data["architecture"]
                if not all(isinstance(n, int) and n > 0 for n in arch):
                    raise ValueError("Invalid architecture: must contain positive integers")
                
                # Create new neural network with loaded architecture
                nn = NeuralNetwork(*arch)
                nn.weight = data["weights"]
                nn.bias = data["biases"]
                
                # Load learning rate if present, otherwise keep default
                nn.learning_rate = data.get("learning_rate", nn.learning_rate)
                
                # Validate weights and biases shapes
                for i in range(len(arch) - 1):
                    expected_weight_shape = (arch[i + 1], arch[i])
                    expected_bias_shape = (arch[i + 1], 1)
                    if (not isinstance(nn.weight[i], np.ndarray) or 
                        nn.weight[i].shape != expected_weight_shape or
                        not isinstance(nn.bias[i], np.ndarray) or 
                        nn.bias[i].shape != expected_bias_shape):
                        raise ValueError(f"Invalid shape for weights or biases at layer {i}")
                
                print(f"Model successfully loaded from {filename}")
                return nn
        except Exception as e:
            raise Exception(f"Error loading model from {filename}: {str(e)}")