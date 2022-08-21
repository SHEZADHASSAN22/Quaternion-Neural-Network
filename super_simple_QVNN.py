import quaternion
import numpy as np

'''This is a super simple quaternion neural network which has only three neurons. 
The network takes in a single fixed input and its goal is to learn the correct weights and biases so as to output
a pre-defined desired output. '''

# Weighted input is the activation of a neuron without the sigmoid fucntion applied
def getWeightedInput(prevActivation, weight, bias):
  return prevActivation * weight + bias

def sigmoid(z):
  return 1.0/(1.0+np.exp(-z))

# Applies activation function(sigmoid) on each component of given quaternion.
def activateNeuron(a, w, b):
  q = getWeightedInput(a, w, b)
  return np.quaternion(sigmoid(q.w), sigmoid(q.x), sigmoid(q.y), sigmoid(q.z))

# Takes in input quaternion, feeds it all the way forward and returns the output.
def feedForward(a1):
  a2 = activateNeuron(a1, w1, b1)
  return activateNeuron(a2, w2, b2)

def getError(output, desired):
  diff = desired - output
  temp = np.square(diff.w) + np.square(diff.x) + np.square(diff.y) + np.square(diff.z)
  return 0.5*(np.square(np.sqrt(temp)))


# Input quaternion.
a1 = np.quaternion(1,1,1,1)

# Desired output quaternion.
k = np.quaternion(0.9,0.5,0.5,0.2)

#learning rate
lr = 0.01
epochs = 70000

# Randomised weights and biases.
w1 = np.quaternion(np.random.rand(),np.random.rand(),np.random.rand(),np.random.rand())
w2 = np.quaternion(np.random.rand(),np.random.rand(),np.random.rand(),np.random.rand())
b1 = np.quaternion(np.random.rand(),np.random.rand(),np.random.rand(),np.random.rand())
b2 = np.quaternion(np.random.rand(),np.random.rand(),np.random.rand(),np.random.rand())

a2 = activateNeuron(a1, w1, b1)
output = feedForward(a1)
print("Output before training: ", output)
print("Error before trainig: ", getError(output, k), "\n")

# Backpropagation
for i in range(epochs):
  a2 = activateNeuron(a1, w1, b1)
  output = feedForward(a1)

  dcdb2 = lr*np.quaternion((output - k).w * (1 - output.w) * output.w,
                        (output - k).x * (1 - output.x) * output.x,
                        (output - k).y * (1 - output.y) * output.y,
                        (output - k).z * (1 - output.z) * output.z)
  dcdw2 = a2 * dcdb2

  dcda1 = dcdb2 * w2

  dcdb1 = np.quaternion((1 - a2.w)*a2.w*dcda1.w,
                        (1 - a2.x)*a2.x*dcda1.x,
                        (1 - a2.y)*a2.y*dcda1.y,
                        (1 - a2.z)*a2.z*dcda1.z)
  dcdw1 = a1 * dcdb1

  w2 = w2 + (dcdw2)
  b2 = b2 + (dcdb2)
  w1 = w1 + (dcdw1)
  b1 = b1 + (dcdb1)


output = feedForward(a1)
print("Output after training: ", output) 
print("Error after training: ", getError(output, k))