import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from torch.utils.data import DataLoader
from tqdm import tqdm


# evaluate the model on the given set
def validate(model: nn.Sequential, device: torch.device, data_loader: DataLoader, criterion):
    sum_loss, sum_correct = 0, 0
    margin = torch.Tensor([]).to(device)

    # switch to evaluation mode
    model.eval()
    with torch.no_grad():
        for i, (data, target) in enumerate(data_loader):
            data, target = data.to(device).view(data.size(0), -1), target.to(device)

            # compute the output
            output = model(data)

            # compute the classification error and loss
            pred = output.max(1)[1]
            sum_correct += pred.eq(target).sum().item()
            sum_loss += len(data) * criterion(output, target).item()

    return (sum_correct / len(data_loader.dataset)), sum_loss / len(data_loader.dataset)


def prepare_data(data: pd.DataFrame, train_size: float = 0.8):
    X = torch.tensor(data.loc[:, ["x_i1", "x_i2"]].values).float()
    Y = torch.tensor(data.l_i.values)
    dataset = torch.utils.data.TensorDataset(X, Y)
    train_size = round(X.shape[0] * train_size)
    test_size = X.shape[0] - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, (train_size, test_size)
    )
    return train_dataset, test_dataset


def train(model: nn.Sequential, device: torch.device, train_loader: DataLoader, criterion,
          optimizer, epoch: int):
    sum_loss, sum_correct = 0, 0
    # switch to train mode
    model.train()
    for i, (data, target) in enumerate(train_loader):
        data, target = data.to(device).view(data.size(0), -1), target.to(device)

        # compute the output
        output = model(data)

        # compute the classification error and loss
        loss = criterion(output, target)
        pred = output.max(1)[1]
        sum_correct += pred.eq(target).sum().item()
        sum_loss += len(data) * loss.item()

        # compute the gradient
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return (sum_correct / len(train_loader.dataset)), sum_loss / len(train_loader.dataset)


def main():
    data_a = pd.read_excel('./data/trainingdata_a.xls')
    train_dataset, test_dataset = prepare_data(data_a)

    # Create simple network
    neurons = 8
    inputs = 2
    classes = 2
    model = nn.Sequential(
        nn.Linear(inputs, neurons),
        nn.ReLU(),
        nn.Linear(neurons, classes),
    )

    device = torch.device("cpu")
    train_loader = torch.utils.data.DataLoader(train_dataset)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    model_path = "model.torch"
    epochs = 100

    # Train network / load already trained network
    try:
        model = torch.load(model_path)
    except FileNotFoundError:
        pbar = tqdm(range(epochs))
        training_accuracy, training_loss = 0, 0
        for epoch in pbar:
            pbar.set_description(
                "(Loss: {:.6f}, Accuracy: {:.4f}) - Progress: ".format(training_loss, training_accuracy))
            training_accuracy, training_loss = train(model, device, train_loader, criterion, optimizer,
                                                     epoch)
        torch.save(model, "model.torch")

    # Check test accuracy
    test_loader = torch.utils.data.DataLoader(test_dataset)
    test_acc, test_loss = validate(model, device, test_loader, criterion)
    print("Test Acc: {:.2f}, Test Loss: {:.2f}".format(test_acc, test_loss))

    # Evaluate boundaries
    data, target = next(iter(train_loader))
    model = BoundedModule(model, data)
    # Define perturbation
    ptb = PerturbationLpNorm(norm=np.inf, eps=0.1)
    # Make the input a BoundedTensor with perturbation
    my_input = BoundedTensor(data, ptb)
    # Regular forward propagation using BoundedTensor works as usual.
    prediction = model(my_input)
    # Compute LiRPA bounds
    lb, ub = model.compute_bounds(x=(my_input,), method="backward")
    print("lower bound:", lb.data)
    print("upper bound:", ub.data)
    print("done")


if __name__ == '__main__':
    main()
