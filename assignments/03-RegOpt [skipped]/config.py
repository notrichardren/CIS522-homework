from typing import Callable
import torch
import torch.optim
import torch.nn as nn
from torchvision.transforms import Compose, Normalize, ToTensor


class CONFIG:
    batch_size = 64
    num_epochs = 4
    initial_learning_rate = 0.05  # Highest learning rate you want.
    final_learning_rate = 0.003  # Lowest learning rate you want.
    initial_weight_decay = 0

    lrs_kwargs = {
        "num_epochs": num_epochs,
        "initial_learning_rate": initial_learning_rate,
        "final_learning_rate": final_learning_rate,
    }

    optimizer_factory: Callable[
        [nn.Module], torch.optim.Optimizer
    ] = lambda model: torch.optim.Adam(
        model.parameters(),
        lr=CONFIG.initial_learning_rate,
        weight_decay=CONFIG.initial_weight_decay,
    )

    transforms = Compose(
        [
            ToTensor(),
        ]
    )
