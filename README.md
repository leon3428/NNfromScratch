# NNfromScratch

Understanding how neural networks work and learn can be quite challenging, to say the least. I find that the best way to learn and truly understand something is to make it from scratch so that is exactly what I set out to do. I decided to use python and numpy to create a dense neural network(including gradient descent and backpropagation) and train it on the mnist dataset. Making one from scratch was challenging but in a good way. It was actually easier than I thought it would be.

## installation
create a virtual environment in the project folder
```
python -m venv venv
```
activate it
```
source venv/bin/activate
```
install numpy
```
pip install numpy
```
and run the main python file
```
python main.py
```

## usage
1) load your data as a python list
2) create a network
```
net = network.Network([784, 30, 10])
```
The list describes how many neurons should each layer have starting from the input layer

3) use stochastic gradient descent to train the network

```
net.SGD(train_data, 30, 20, 0.1, test_data=test_data)
```
Parameters are training_data, epochs, batch size, learning rate, and test_data

## results

With 30 hidden neurons and 10 output neurons, I was able to get an accuracy of 94.6% on the mnist dataset. Without modern optimizers, the network often gets stuck in a local minimum so this is the best accuracy out of three runs.

## resources
[3Blue1Brown, neural networks series]( https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)

[Michael Nielsen, Neural networks and deep learining]( http://neuralnetworksanddeeplearning.com/index.html)
