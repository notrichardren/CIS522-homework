import torch

# My model architecture is heavily based on LeNet and the PyTorch tutorial for CNN.
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# https://en.wikipedia.org/wiki/LeNet


class Model(torch.nn.Module):
    def __init__(self, num_channels: int, num_classes: int) -> None:
        super().__init__()

        # (32, 32, 3)
        self.conv1 = torch.nn.Conv2d(
            in_channels=num_channels,
            out_channels=10,
            kernel_size=5,
            padding=2,
            stride=1,
        )
        # (32, 32, 3)
        self.batchnorm1 = torch.nn.BatchNorm2d(10)
        self.relu1 = torch.nn.ReLU()
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        # (16, 16, 10)

        # (16, 16, 10)
        self.conv2 = torch.nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)
        # (12, 12, 20)
        self.batchnorm2 = torch.nn.BatchNorm2d(20)
        self.relu2 = torch.nn.ReLU()
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        # (6, 6, 20)

        self.fc1 = torch.nn.Linear(720, 120)
        self.relu3 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(120, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.batchnorm1(x) #10, 32, 32
        x = self.relu1(x)
        x = self.maxpool1(x) #10, 16, 16

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu2(x)
        x = self.maxpool2(x) #

        x = torch.flatten(x, start_dim = 1) #20, 6, 6

        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x


# I asked ChatGPT 3.5 to give some general tips for now to make a good-performing model. I did not use any code from ChatGPT.

"""
I can give you some general tips on how to design a neural network architecture that can train quickly and efficiently on image classification tasks:

Use small kernels: Smaller kernels (e.g. 3x3 or 5x5) can be more efficient than larger kernels, as they require fewer parameters and can capture local features more effectively. However, you might need to use multiple layers to capture more complex patterns.

Use max pooling: Max pooling can be used to downsample the feature maps and reduce the number of parameters in the network. However, it can also result in some loss of information.

Use batch normalization: Batch normalization can help stabilize the training process and reduce the number of epochs required to reach convergence.

Use skip connections: Skip connections can help improve the gradient flow during training and prevent the vanishing gradient problem. They can also help reduce the number of parameters required in the network.

Use data augmentation: Data augmentation can help increase the amount of training data and improve the generalization performance of the network.

Use early stopping: Early stopping can help prevent overfitting and reduce the training time required to reach a good level of performance.
"""
