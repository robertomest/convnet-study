mkdir -p cifar10
#CIFAR-10 dataset
wget -O cifar10/cifar.tar.gz http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xzvf cifar10/cifar.tar.gz -C cifar10 --strip-components=1
rm cifar10/cifar.tar.gz

#CIFAR-100 dataset
mkdir -p cifar100
wget -O cifar100/cifar100.tar.gz https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
tar -xzvf cifar100/cifar100.tar.gz -C cifar100 --strip-components=1
rm cifar100/cifar100.tar.gz

mkdir -p mnist
#MNIST dataset
wget -O mnist/train-images-idx3-ubyte.gz http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget -O mnist/train-labels-idx1-ubyte.gz http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget -O mnist/t10k-images-idx3-ubyte.gz http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget -O mnist/t10k-labels-idx1-ubyte.gz http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

mkdir -p svhn
#SVHN dataset
wget -O svhn/train_32x32.mat http://ufldl.stanford.edu/housenumbers/train_32x32.mat
wget -O svhn/test_32x32.mat http://ufldl.stanford.edu/housenumbers/test_32x32.mat
wget -O svhn/extra_32x32.mat http://ufldl.stanford.edu/housenumbers/extra_32x32.mat
