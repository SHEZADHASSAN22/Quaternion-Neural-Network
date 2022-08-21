import numpy as np
import quaternion
import random

'''
This code is adapted from Michael Nielsens tutorial on neural networks which can be found:
http://neuralnetworksanddeeplearning.com/chap1.html

The differences between this code and Nielsens are that:
1.  All the real numbers are replaced with quaternions
2. The way that the sigmoid function is applied is slighlty different in that it is applied to each componenet of the quaternion
3. The back-propagation formulas are different

It may be wise to first study his code to understand the logic behind the code and then come back to this and look
at the differences made. 
'''

class Network:
    def __init__(self, size) -> None:
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly."""
        self.size = size
        self.layers = len(size)
        self.biases = [quaternion.as_quat_array(np.random.rand(y, 1, 4)) for y in size[1:]]
        self.weights = [quaternion.as_quat_array(np.random.rand(y, x, 4))
                        for x, y in zip(size[:-1], size[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input. 
        Note that the sigmoid function is applied to each individual component of the quaternion. """
        for b, w in zip(self.biases, self.weights):
            #a = sigmoid(dot(w, a)+b)           # Is this better?
            a = quatDot(a.transpose(), w.transpose()).transpose()+b
            for i in range(len(a)):
                a[i][0] = np.quaternion(sigmoid(a[i][0].w), sigmoid(a[i][0].x),sigmoid(a[i][0].y),sigmoid(a[i][0].z))
        return a

    def SGD(self, training_data, epochs, lr, mini_batch_size):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs. """
        n = len(training_data)

        for j in range(epochs):
            #print("epoch ", j)

            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, lr)

    def update_mini_batch(self, mini_batch, lr):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        # Here I am basically copying the shape of the bias matrix and for each bias adding an array with 
        # 4 elements so that I can then conver that into a quternion
        nabla_b = [quaternion.as_quat_array(np.zeros(np.append(np.asarray(b.shape),[4]))) for b in self.biases]
        nabla_w = [quaternion.as_quat_array(np.zeros(np.append(np.asarray(w.shape),[4]))) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y, lr)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w + (nw/len(mini_batch))
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b + (nb/len(mini_batch))
                       for b, nb in zip(self.biases, nabla_b)]


    def backprop(self, x, y, lr):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``. The backprop formulas were
        taken from the paper titled 'A Quaternary version of the backpropogation
        algorithm' by Tohru Nitta (https://www2.rikkyo.ac.jp/web/tnitta/icnn95.pdf)."""
        nabla_b = [quaternion.as_quat_array(np.zeros(np.append(np.asarray(b.shape),[4]))) for b in self.biases]
        nabla_w = [quaternion.as_quat_array(np.zeros(np.append(np.asarray(w.shape),[4]))) for w in self.weights]

        activation = x
        activations = [x] # list to store all the activations, layer by layer

        for b, w in zip(self.biases, self.weights):
            z = quatDot(activation.transpose(), w.transpose()).transpose()+b

            shape = np.append(np.asarray(z.shape), [4])
            activation = quaternion.as_quat_array(np.zeros(shape))

            # Apply sigmoid fucntion
            for i in range(len(z)):
                activation[i] = np.quaternion(sigmoid(z[i][0].w), sigmoid(z[i][0].x),sigmoid(z[i][0].y),sigmoid(z[i][0].z))
            activations.append(activation)

        # dc/db for output layer
        shape = np.append(np.asarray(self.biases[-1].shape), [4])
        delta = quaternion.as_quat_array(np.zeros(shape))
        
        for i in range(len(nabla_b[-1])):
            k = y[i][0]
            output = activations[-1][i][0]
            delta[i] = np.array([lr*np.quaternion((output - k).w * (1 - output.w) * output.w, 
                                                    (output - k).x * (1 - output.x) * output.x,
                                                    (output - k).y * (1 - output.y) * output.y,
                                                    (output - k).z * (1 - output.z) * output.z)])
        # dc/dw for output layer
        nabla_b[-1] = delta
        nabla_w[-1] = quatDot(activations[-2], delta.transpose()).transpose()

        for l in range(2, self.layers):
            dcda = quatDot(delta.transpose(), self.weights[-l+1]).transpose()
            shape = np.append(np.asarray(self.biases[-l].shape), [4])
            delta = quaternion.as_quat_array(np.zeros(shape))

            for i in range(len(nabla_b[-l])):
                a2 = activations[-l][i][0]
                dcda_curr = dcda[i][0]
                delta[i] = np.array([np.quaternion((1 - a2.w) * (a2.w) * dcda_curr.w, 
                                                    (1 - a2.x) * (a2.x) * dcda_curr.x,
                                                    (1 - a2.y) * (a2.y) * dcda_curr.y,
                                                    (1 - a2.z) * (a2.z) * dcda_curr.z)])

            nabla_b[-l] = delta
            nabla_w[-l] = quatDot(activations[-l-1], delta.transpose()).transpose()
        
        return (nabla_b, nabla_w) 
            
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def arraySum(a):
    """Sums up the elements of an array with quaternion values."""
    temp = np.quaternion(0,0,0,0)
    for i in a:
        temp = temp + i
    return temp

def quatDot(l, r):
    """The quaternion version of the dot product."""
    depth = len(l)
    width = len(r[0])

    if(len(l[0]) != len(r)):
        raise Exception("invalid dimensions")

    res = quaternion.as_quat_array(np.zeros([depth, width, 4]))
    r = r.transpose()

    for i in range(depth):
        for j in range(width):
            res[i][j] = arraySum(l[i] * r[j])
    
    return res

def funcToSimulate(x):
    """The function that the neural network is trying to simulate."""
    return x

def generateData(qty, func=funcToSimulate):
    """Generates the data to train the network using function described in funcToSimulate."""
    res = []
    # i and func(i) are in arrays cause further functions may have multiple inputs and outputs
    for i in range(qty):
        i = i/1000
        input = np.array([[np.quaternion(i,i,i,i)]])
        output = np.array([[func(np.quaternion(i,i,i,i))]])
        res.append((input, output))
    return res

def getError(output, desired):
  diff = desired - output
  temp = np.square(diff.w) + np.square(diff.x) + np.square(diff.y) + np.square(diff.z)
  return 0.5*(np.square(np.sqrt(temp)))

# Initialise the network and generate the data
network = Network([1, 16, 1])
data = generateData(1000)

# Print the output and error before training
og_output = network.feedforward(np.array([[np.quaternion(1, 1, 1, 1)]]))
print("Original output: ", og_output[0][0])
print("Error: ", getError(og_output[0][0], data[0][1][0][0]))
print()

# Train the model using stochastic gradient descent
network.SGD(data, 5, 0.01, 1)

# Print output and error after training
output = network.feedforward(np.array([[np.quaternion(1, 1, 1, 1)]]))
print("Output: ", output[0][0])
print("Error: ", getError(output[0][0],data[0][1][0][0]))
