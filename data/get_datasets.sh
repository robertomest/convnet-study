mkdir -p cifar10
#CIFAR-10 dataset
wget -O cifar10/cifar.tar.gz http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xzvf cifar10/cifar.tar.gz -C cifar10 --strip-components=1
rm cifar10/cifar.tar.gz

mkdir -p mnist
#MNIST dataset
wget -O mnist/train-images-idx3-ubyte.gz http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget -O mnist/train-labels-idx1-ubyte.gz http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget -O mnist/t10k-images-idx3-ubyte.gz http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget -O mnist/t10k-labels-idx1-ubyte.gz http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
