import math
import random

import torch
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torchvision.transforms.functional as TF
from torchvision.io import read_image, ImageReadMode

torch.manual_seed(1)
# current_dataset = "EMNIST"
# Model Configuration
batch_size = 128
input_image_height = 28
kernel_size = 3
conv_output_size = (input_image_height - kernel_size + 1) * (input_image_height - kernel_size + 1)
hidden_size = 1600
# global variables so I only have to initialize them once.
cached_testing_data = None
cached_training_data = None
active_model = None
accessor = {
    'model_name': 'new_model_name',
    'accuracy': 0.0,
    'loss': 0,
    'trainable_params': 0,
    'dataset': 'IDFK',
    'device': 'i made it up'
}


def get_device():
    # get CPU, GPU, or MPS device for training/running model
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    return device


def get_model_instance(device, dataset):
    return NeuralNetwork(dataset).to(device)


def get_model(device, dataset, model_name):
    #model = get_model_instance(device, dataset)
    model = torch.load(model_name)
    #model.load_state_dict(torch.load(model_name))
    return model


def evaluate(model_name):
    device = get_device()
    dataset = get_dataset_from_model_name(model_name)
    model = get_model(device, dataset, model_name)  # model gets loaded here
    testing_dataloader = DataLoader(get_testing_data(dataset), batch_size=batch_size)
    loss_functn = nn.CrossEntropyLoss()
    accuracy, loss = test(testing_dataloader, model, loss_functn, device)
    trainable_params = count_parameters(model)

    accessor['accuracy'] = accuracy
    accessor['loss'] = loss
    accessor['trainable_params'] = trainable_params
    accessor['dataset'] = dataset
    accessor['device'] = device
    accessor['model_name'] = model_name
    return model


def get_training_data(dataset_str):
    training_data = None
    if dataset_str == 'MNIST':
        training_data = datasets.MNIST(
            root="data",
            train=True,
            download=True,
            transform=transforms.ToTensor()
        )
    elif dataset_str == 'EMNIST':
        transform = transforms.Compose([
            # flip horizontally
            # rotate 90 degrees
            lambda img: TF.rotate(img, -90),
            lambda img: TF.hflip(img),
            transforms.ToTensor()
        ])
        training_data = datasets.EMNIST(
            root="data",
            train=True,
            download=True,
            transform=transform,
            split="bymerge"
        )
    else:
        print("invalid dataset. Must either be MNIST or EMNIST")
    global cached_training_data
    cached_training_data = training_data
    return training_data


def get_testing_data(dataset_str):
    testing_data = None
    if dataset_str == 'MNIST':
        testing_data = datasets.MNIST(
            root="data",
            train=False,
            download=True,
            transform=transforms.ToTensor()
        )
    elif dataset_str == 'EMNIST':
        transform = transforms.Compose([
            # flip horizontally
            # rotate 90 degrees
            lambda img: TF.rotate(img, -90),
            lambda img: TF.hflip(img),
            transforms.ToTensor()
        ])
        testing_data = datasets.EMNIST(
            root="data",
            train=False,
            download=True,
            transform=transform,
            split="bymerge"
        )
    else:
        print("invalid dataset. Must either be MNIST or EMNIST")
    global cached_testing_data
    cached_testing_data = testing_data
    return testing_data


def get_num_classes(dataset_str):
    if dataset_str == 'MNIST':
        return 10
    elif dataset_str == 'EMNIST':
        return 47


def get_classes(dataset_str):
    if dataset_str == 'MNIST':
        return "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"
    elif dataset_str == 'EMNIST':
        return ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
                "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
                "U", "V", "W", "X", "Y", "Z", "a", "b", "d", "e",
                "f", "g", "h", "n", "q", "r", "t"
                )


def predictions_layer_to_list(nn_output, dataset):
    probabilities = list()  # list of tuples
    # nn_index is a [1, 47] tensor
    outputs_list = nn_output[0].tolist()
    for index, probability in enumerate(outputs_list):
        character = get_classes(dataset)[index]
        probabilities.append((probability, character))
    return probabilities


def predictions_layer_to_sorted_list(nn_output, dataset, count=10):
    # selection sort, but we remove instead of swap and we don't finish
    probabilities = predictions_layer_to_list(nn_output, dataset)
    unsorted = tuple(probabilities)
    sorted_probabilities = list()
    for i in range(count):
        highest = -math.inf
        highest_index = -1
        for index, (prob, _) in enumerate(unsorted):
            if prob > highest:
                highest_index = index
                highest = prob
        sorted_probabilities.append(unsorted[highest_index])
        unsorted = list_without_index(unsorted, highest_index)
    return sorted_probabilities


def list_without_index(in_list, index):
    if index < 0:
        return in_list
    out_list = in_list[:index]
    out_list += in_list[index + 1:]
    return out_list


def tensor_to_png(tensor, filename):
    transform = transforms.ToPILImage()
    img = transform(tensor)
    img.save(filename)


def test_index_to_png(index, filename, dataset):
    input_tensor = get_testing_data(dataset)[index][0]
    tensor_to_png(input_tensor, filename)


def classify_image(image, dataset, model_name):
    device = get_device()
    model = get_model(device, dataset, model_name)
    image_tensor = read_image(image, ImageReadMode.GRAY)
    image_tensor = image_tensor.unsqueeze(1)
    image_tensor = image_tensor.float()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        predictions = model(image_tensor)
    return predictions_layer_to_sorted_list(predictions, dataset)


def classify_training_image_index(index, dataset, model_name, filename):
    device = get_device()
    testing_data = get_testing_data(dataset)
    input_tensor = testing_data[index][0]
    input_tensor = input_tensor.unsqueeze(1)
    model = get_model(device, dataset, model_name)  # make sure this loads the state dict or else you get an
    # untrained model
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        predictions = model(input_tensor)
    test_index_to_png(index, filename, dataset)
    return predictions_layer_to_sorted_list(predictions, dataset)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    # .numel() returns the total number of elements in a tensor


def get_test_image_label(index, dataset):
    label_index = get_testing_data(dataset)[index][1]
    return get_classes(dataset)[label_index]


def get_dataset_from_model_name(model_name):
    return model_name.split("_")[0]


class NeuralNetwork(nn.Module):  # inherits from nn.Module
    def __init__(self, dataset):
        super().__init__()
        self.flatten = nn.Flatten()
        self.sequential_network = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size, bias=False),
            nn.Flatten(),  # since the convolution spits out 26x26 sub-images.
            nn.Linear(conv_output_size, hidden_size, bias=False),
            # the 3x3 convolution reduces the number of pixels by 2.
            nn.Tanh(),
            nn.Dropout(p=0.5),  # prevents co-adaptation of feature detectors preventing overfitting.
            nn.Linear(hidden_size, get_num_classes(dataset), bias=False),
        )

    def forward(self, input_features):
        # flattened_input = self.flatten(input_features)
        network_output = self.sequential_network(input_features)  # this runs the input features through all the layers.
        return network_output


def train(dataloader, model, loss_function, optimizer, device):
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


def test(dataloader, model, loss_function, device):
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
    accuracy = (100 * correct)
    print(f"Test Error: \n Accuracy: {accuracy:>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return accuracy, test_loss


def main(model_name):
    # get CPU, GPU, or MPS device for training
    device = get_device()
    dataset = get_dataset_from_model_name(model_name)
    model = get_model_instance(device, dataset)
    classes = get_classes(dataset)
    # Create data loaders.
    testing_data = get_testing_data(dataset)
    training_data = get_training_data(dataset)
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(testing_data, batch_size=batch_size)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)  # lr stands for learning rate
    scheduler = ExponentialLR(optimizer, gamma=0.9)  # makes the learning rate slow down with each epoch

    if TRAINING_NEW_MODEL:
        epochs = 5
        for t in range(epochs):
            print(f"Epoch {t + 1}\n-----------------------------")
            train(train_dataloader, model, loss_function, optimizer, device)
            test(test_dataloader, model, loss_function, device)
        print("Done!")

        # Save model
        torch.save(model, new_model_name)
        print(f"Saved Pytorch Model State to {new_model_name}")
    else:
        # Load existing model
        model = evaluate(model_name)
    model.eval()
    final_test_index = random.randint(0, 100)
    x, y = testing_data[final_test_index][0], testing_data[final_test_index][1]
    x = x.unsqueeze(1)
    with torch.no_grad():
        x = x.to(device)
        pred = model(x)
        predicted_index = pred[0].argmax(0)
        predicted, actual = classes[predicted_index], classes[y]
        confidence = pred[0][predicted_index].item()
        print(f'Predicted: "{predicted}", Actual: "{actual}", Confidence: "{confidence}"')
        print(f"Using {count_parameters(model)} parameters")


TRAINING_NEW_MODEL = True
new_model_name = "EMNIST_1.pth"  # should be named '[dataset]_[usually a number].pth'
if __name__ == "__main__":
    main(new_model_name)
