# get mnist data
from torchvision import datasets
def get_mnist_data():
    data_dir = './data'

    flatten = False
    mnist_train_set = datasets.MNIST(data_dir + '/mnist/', train = True, download = True)
    mnist_test_set = datasets.MNIST(data_dir + '/mnist/', train = False, download = True)

    train_input = mnist_train_set.data.view(-1, 1, 28, 28).float()
    train_target = mnist_train_set.targets
    test_input = mnist_test_set.data.view(-1, 1, 28, 28).float()
    test_target = mnist_test_set.targets

    if flatten:
        train_input = train_input.clone().reshape(train_input.size(0), -1)
        test_input = test_input.clone().reshape(test_input.size(0), -1)
    return train_input, test_input, train_target, test_target