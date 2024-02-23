import math
import random

import torch
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

torch.manual_seed(1)
TRAINING_NEW_MODEL = False
model_name = "character_classifier_model"

batch_size = 128
input_image_height = 28
kernel_size = 5
conv_output_size = (input_image_height - kernel_size + 1) * (input_image_height - kernel_size + 1)
hidden_size = 800
num_classes = 47


# hidden_size = math.ceil((conv_output_size + 10) / 2)


class NeuralNetwork(nn.Module):  # inherits from nn.Module
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.sequential_network = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size, bias=False),
            nn.Flatten(),  # since the convolution spits out 26x26 sub-images.
            nn.Linear(conv_output_size, hidden_size, bias=False), # the 3x3 convolution reduces the number of pixels by 2.
            nn.Tanh(),
            nn.Dropout(p=0.5),  # prevents co-adaptation of feature detectors preventing overfitting.
            nn.Linear(hidden_size, num_classes, bias=False),
        )

    def forward(self, input_features):
        # flattened_input = self.flatten(input_features)
        network_output = self.sequential_network(input_features)  # this runs the input features through all the layers.
        return network_output


def train(dataloader, model, loss_function, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (input_image, label) in enumerate(dataloader):  # input_image often referred to as X, and label as y
        input_image, label = input_image.to(device), label.to(device)
        # Compute prediction error
        prediction = model(input_image)
        loss = loss_function(prediction, label)

        # Backpropagation
        loss.backward()
        optimizer.step()  # right here is where the weights get updated
        optimizer.zero_grad()  # sets the gradients to zero in preparation for the next iteration.
        # Normally pytorch averages out the gradient with each iteration.

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(input_image)  # .item() gets the first item in a tensor.
            # Usually this is used when there is only one item in a tensor
            print(
                f"loss: {loss:>7f} [{current:>5d}/{size:>5d}")


def test(dataloader, model, loss_function):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():  # 'with' enters the torch.no_grad() context which
        # is used to temporarily disable gradient computation.
        for input_image, label in dataloader:
            input_image, label = input_image.to(device), label.to(device)
            prediction = model(input_image)
            test_loss += loss_function(prediction, label).item()
            correct += (prediction.argmax(1) == label).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def count_parameters():
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    # .numel() returns the total number of elements in a tensor


if __name__ == "__main__":
    # get CPU, GPU, or MPS device for training
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    model = NeuralNetwork().to(device)

    # Download training data from open datasets.
    test_data = datasets.EMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
        split="bymerge"
    )
    training_data = datasets.EMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
        split="bymerge"
    )

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)  # lr stands for learning rate
    scheduler = ExponentialLR(optimizer, gamma=0.9)  # makes the learning rate slow down with each epoch

    if TRAINING_NEW_MODEL:
        epochs = 5
        for t in range(epochs):
            print(f"Epoch {t + 1}\n-----------------------------")
            train(train_dataloader, model, loss_function, optimizer)
            test(test_dataloader, model, loss_function)
        print("Done!")

        # Save model
        torch.save(model.state_dict(), f"{model_name}.pth")
        print(f"Saved Pytorch Model State to {model_name}.pth")
    else:
        # Load existing model
        model = NeuralNetwork().to(device)
        model.load_state_dict(torch.load(f"{model_name}.pth"))
        test(test_dataloader, model, loss_function)

    classes = [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "J",
        "K",
        "L",
        "M",
        "N",
        "O",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "U",
        "V",
        "W",
        "X",
        "Y",
        "Z",
        "a",
        "b",
        "d",
        "e",
        "f",
        "g",
        "h",
        "n",
        "q",
        "r",
        "t"
    ]

    model.eval()
    final_test_index = random.randint(0, 100)
    x, y = test_data[final_test_index][0], test_data[final_test_index][1]
    with torch.no_grad():
        x = x.to(device)
        pred = model(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')
        print(f"Using {count_parameters()} parameters")
