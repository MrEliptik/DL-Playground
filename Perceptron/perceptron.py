import numpy as np

class Perceptron():
    def __init__(self):
        self.syn_weights = np.random.rand(4, 1)
    
    # Activation function
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Derivative used for back propagation
    def sigmoid_deriv(self, x):
        return np.exp(-x)/((1 + np.exp(-x))**2)

    def train(self, inputs, real_outputs, its, lr):
        delta_weights = np.zeros((4, 7))
        for iteration in (range(its)):
            # Forward Pass
            z = np.dot(inputs, self.syn_weights)
            activation = self.sigmoid(z)
            # Backward Pass
            for i in range(7):
                cost = (activation[i] - real_outputs[i])**2
                cost_prime = 2*(activation[i] - real_outputs[i])
                for n in range(4):
                    delta_weights[n][i] = cost_prime * \
                        inputs[i][n] * self.sigmoid_deriv(z[i])
        delta_avg = np.array([np.average(delta_weights, axis=1)]).T
        self.syn_weights = self.syn_weights - delta_avg*lr

    def results(self, inputs):
        return self.sigmoid(np.dot(inputs, self.syn_weights))


if __name__ == "__main__":
    # Create training sets
    ts_input = np.array([[0, 0, 1, 0],
                         [1, 1, 1, 0],
                         [1, 0, 1, 1],
                         [0, 1, 1, 1],
                         [0, 1, 0, 1],
                         [1, 1, 1, 1],
                         [0, 0, 0, 0]])
    ts_output = np.array([[0, 1, 1, 0, 0, 1, 0]]).T
    testing_data = np.array([[0, 1, 1, 0],
                             [0, 0, 0, 1],
                             [0, 1, 0, 0],
                             [1, 0, 0, 1],
                             [1, 0, 0, 0],
                             [1, 1, 0, 0],
                             [1, 0, 1, 0]])
    lr = 10  # learning rate
    steps = 10000
    perceptron = Perceptron()  # initialize a perceptron
    perceptron.train(ts_input, ts_output, steps, lr)  # train the perceptron
    results = []
    for x in (range(len(testing_data))):
        run = testing_data[x]
        trial = perceptron.results(run)
        results.append(trial.tolist())
    print("results")
    print(results)
    print(np.ravel(np.rint(results)))
    print(perceptron.syn_weights)
