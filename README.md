# Quaternion-Valued-Neural-Network

Thhe purpose of this project was to build a quaternion neural network from scratch using only numpy. 

The super_simple_QVNN is as implied a very simple network consisting of only three neurons. This network has only one set of training data 
which consists of a fixed input and a pre-defined desired output. The goals is to get the network to learn the correct weights ans biases,
all of ehich are quaternions, to output the desired quaternion value using a quaternion version of the back-propagation algorithm. 

The quaternion_NN is an 'quaternified' adaptation of Michael Nielsens neural network which he explains in detail in his online tutorial on 
neural networks(http://neuralnetworksanddeeplearning.com/chap1.html). The goal of this network was to emulate a specified function 
which was sused to generate the training data. The idea is the same as the super_simple version but expanded to take in more layers and neurons. 
