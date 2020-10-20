import mnist_loader as ml
import network

train_data, val_data, test_data = map(list,ml.load_data_wrapper())

print(len(train_data), len(test_data))
net = network.Network([784, 30, 10])
net.SGD(train_data, 30, 20, 0.1, test_data=test_data)
