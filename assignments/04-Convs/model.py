import torch

# My model architecture is heavily based on LeNet and the PyTorch tutorial for CNN.
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# https://en.wikipedia.org/wiki/LeNet

# 35.5 seconds with 55% accuracy.

batch_size = 100
num_epochs = 8


class Model(torch.nn.Module):
    def __init__(self, num_channels: int, num_classes: int) -> None:
        super().__init__()
        """
        Model
        """

        # (32, 32, 3)
        self.conv1 = torch.nn.Conv2d(
            in_channels=num_channels,
            out_channels=32,
            kernel_size=3,
            padding=1,
            stride=1,
        )
        # (32, 32, 8)
        self.relu1 = torch.nn.ReLU()
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        # (16, 16, 8)

        # (16, 16, 8)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        # (14, 14, 16)
        self.relu2 = torch.nn.ReLU()
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        # (7, 7, 16)
        self.fc1 = torch.nn.Linear(7 * 7 * 64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward function
        """

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = torch.flatten(x, start_dim=1)

        x = self.fc1(x)
        return x


# I asked ChatGPT 3.5 to give some general tips for now to make a good-performing model. I did not
# use any code from ChatGPT.

"""
I can give you some general tips on how to design a neural network architecture that can train 
quickly and efficiently on image classification tasks:

Use small kernels: Smaller kernels (e.g. 3x3 or 5x5) can be more efficient than larger kernels, as 
they require fewer parameters and can capture local features more effectively. However, you might \
need to use multiple layers to capture more complex patterns.

Use max pooling: Max pooling can be used to downsample the feature maps and reduce the number of 
parameters in the network. However, it can also result in some loss of information.

Use batch normalization: Batch normalization can help stabilize the training process and reduce the 
number of epochs required to reach convergence.

Use skip connections: Skip connections can help improve the gradient flow during training and 
prevent the vanishing gradient problem. They can also help reduce the number of parameters required 
in the network.

Use data augmentation: Data augmentation can help increase the amount of training data and improve 
the generalization performance of the network.

Use early stopping: Early stopping can help prevent overfitting and reduce the training time 
required to reach a good level of performance.
"""
