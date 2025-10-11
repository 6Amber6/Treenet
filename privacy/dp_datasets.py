import torch

def get_toy_dataset(num_classes=4, n_train=1000, n_test=200):
    # simple Gaussian blobs, 28×28 grayscale
    def make_split(n):
        x = torch.randn(n, 1, 28, 28)
        y = torch.randint(0, num_classes, (n,))
        return x, y
    x_train, y_train = make_split(n_train)
    x_test, y_test = make_split(n_test)
    return x_train, y_train, x_test, y_test
