# DOC: MODIFIED TO DO HYPERPARAMETER TUNING

#%%

"""
Run the MLP training and evaluation pipeline.
"""

# from model_factory import create_model

# MNIST:
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Compose

# PyTorch:
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

from model import MLP

# Other:
from typing import Tuple
from tqdm import tqdm

#%%

# The transform list is a set of operations that we apply to the data
# before we use it. In this case, we convert the data to a tensor and
# flatten it. (Thought-exercise: Why do we need to flatten the data?)
_transform_list = [
    ToTensor(),
    lambda x: x.view(-1),
]


def get_mnist_data() -> Tuple[DataLoader, DataLoader]:
    """
    Get the MNIST data from torchvision.

    Arguments:
        None

    Returns:
        train_loader (DataLoader): The training data loader.
        test_loader (DataLoader): The test data loader.

    """
    # Get the training data:
    train_data = MNIST(
        root="data", train=True, download=True, transform=Compose(_transform_list)
    )
    # Create a data loader for the training data:
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    # Get the test data:
    test_data = MNIST(
        root="data", train=False, download=True, transform=Compose(_transform_list)
    )
    # Create a data loader for the test data:
    test_loader = DataLoader(test_data, batch_size=64, shuffle=True)
    # Return the data loaders:
    return train_loader, test_loader


def train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int,
    learning_rate: float,
    device: torch.device,
) -> int:
    """
    Train a model on the MNIST data.

    Arguments:
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): The training data loader.
        test_loader (DataLoader): The test data loader.
        num_epochs (int): The number of epochs to train for.
        learning_rate (float): The learning rate to use.
        device (torch.device): The device to use for training.

    Returns:
        None

    """
    # Create an optimizer:
    optimizer = Adam(model.parameters(), lr=learning_rate)
    # Create a loss function:
    criterion = CrossEntropyLoss()
    # Move the model to the device:
    model.to(device)
    # Create a progress bar:
    progress_bar = tqdm(range(num_epochs))
    # Train the model:
    for epoch in progress_bar:
        # Set the model to training mode:
        model.train()
        # Iterate over the training data:
        for batch in train_loader:
            # Get the data and labels:
            data, labels = batch
            # Move the data and labels to the device:
            data = data.to(device)
            labels = labels.to(device)
            # Zero the gradients:
            optimizer.zero_grad()
            # Forward pass:
            outputs = model(data)
            # Calculate the loss:
            loss = criterion(outputs, labels)
            # Backward pass:
            loss.backward()
            # Update the parameters:
            optimizer.step()
        # Set the model to evaluation mode:
        model.eval()

        # Calculate the accuracy on the test data:
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in test_loader:
                # Get the data and labels:
                data, labels = batch
                # Move the data and labels to the device:
                data = data.to(device)
                labels = labels.to(device)
                # Forward pass:
                outputs = model(data)
                # Get the predictions:
                _, predictions = torch.max(outputs.data, 1)
                # Update the total and correct counts:
                total += labels.size(0)
                correct += (predictions == labels).sum().item()
        # Calculate the accuracy:
        accuracy = correct / total
        # Update the progress bar:
        progress_bar.set_description(f"Epoch: {epoch}, Accuracy: {accuracy:.4f}")
    return accuracy


def main():

    highPerf = 0
    highPerf_hiddensize = 0
    highPerf_hiddencount = 0

    for hidden_size in torch.arange(20, 300, 60):
        for hidden_count in torch.arange(2, 15, 4):
            # Get the data:
            train_loader, test_loader = get_mnist_data()
            # Create the model:
            model = MLP(
                784, hidden_size, 10, hidden_count, torch.nn.ReLU, torch.nn.init.ones_
            )
            # Train the model:
            accuracy = train(
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                num_epochs=10,
                learning_rate=0.001,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            )
            if highPerf < accuracy:
                accuracy = highPerf
                highPerf_hiddensize = hidden_size
                highPerf_hiddencount = hidden_count

    print(f"Accuracy: {highPerf}")
    print(f"Hidden Size: {hidden_size}")
    print(f"Hidden Count: {hidden_count}")


if __name__ == "__main__":
    main()

# %%

"""
Epoch: 9, Accuracy: 0.9107: 100%|██████████| 10/10 [01:14<00:00,  7.40s/it]
Epoch: 9, Accuracy: 0.1331: 100%|██████████| 10/10 [01:20<00:00,  8.03s/it]
Epoch: 9, Accuracy: 0.1032: 100%|██████████| 10/10 [01:31<00:00,  9.13s/it]
Epoch: 9, Accuracy: 0.0974: 100%|██████████| 10/10 [01:33<00:00,  9.33s/it]
Epoch: 9, Accuracy: 0.9038: 100%|██████████| 10/10 [01:37<00:00,  9.78s/it]
Epoch: 9, Accuracy: 0.0982: 100%|██████████| 10/10 [02:24<00:00, 14.46s/it]
Epoch: 9, Accuracy: 0.1032: 100%|██████████| 10/10 [03:16<00:00, 19.65s/it]
Epoch: 9, Accuracy: 0.0990: 100%|██████████| 10/10 [04:08<00:00, 24.81s/it]
Epoch: 9, Accuracy: 0.9015: 100%|██████████| 10/10 [01:39<00:00,  9.93s/it]
Epoch: 9, Accuracy: 0.1032: 100%|██████████| 10/10 [02:58<00:00, 17.85s/it]
Epoch: 9, Accuracy: 0.0974: 100%|██████████| 10/10 [04:21<00:00, 26.17s/it]
Epoch: 9, Accuracy: 0.0975: 100%|██████████| 10/10 [05:26<00:00, 32.60s/it]
Epoch: 9, Accuracy: 0.8890: 100%|██████████| 10/10 [01:46<00:00, 10.61s/it]
Epoch: 9, Accuracy: 0.1135: 100%|██████████| 10/10 [03:23<00:00, 20.32s/it]
Epoch: 9, Accuracy: 0.1135: 100%|██████████| 10/10 [06:47<00:00, 40.76s/it]
Epoch: 9, Accuracy: 0.0979: 100%|██████████| 10/10 [08:52<00:00, 53.25s/it]
Epoch: 9, Accuracy: 0.8317: 100%|██████████| 10/10 [02:10<00:00, 13.04s/it]
Epoch: 9, Accuracy: 0.1010: 100%|██████████| 10/10 [04:38<00:00, 27.82s/it]
Epoch: 9, Accuracy: 0.1135: 100%|██████████| 10/10 [07:09<00:00, 42.91s/it]
Epoch: 9, Accuracy: 0.1010: 100%|██████████| 10/10 [10:05<00:00, 60.54s/it]
"""
