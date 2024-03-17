import torchvision.datasets as tvds


def download_mnist_train(root: str = "./datasets"):
    tvds.MNIST(root=root, train=True, download=True)


def download_cifar10_train(root: str = "./datasets"):
    tvds.CIFAR10(root=root, train=True, download=True)


# def download_mnist_test(root: str = "./datasets"):
#     tvds.MNIST(root=root, train=False, download=True)


# def download_cifar10_test(root: str = "./datasets"):
#     tvds.CIFAR10(root=root, train=False, download=True)


if __name__ == "__main__":
    print("Downloading CIFAR10-train")
    # print("Downloading CIFAR10-test")
    # download_mnist_train()
    # download_mnist_test()
    download_cifar10_train()
    # download_cifar10_test()

