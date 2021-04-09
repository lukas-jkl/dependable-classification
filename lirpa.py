import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt

# evaluate the model on the given set
def test(model: nn.Sequential, device: torch.device, data_loader: DataLoader, criterion) -> (float, float):
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


def prepare_data(data: pd.DataFrame, train_size: float = 0.8) -> (Dataset, Dataset):
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
          optimizer, epoch: int) -> (float, float):
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


def plot_results(ub_predictions, lb_predictions, predictions, data_loader, epsilon, log_name):
    data, targets = data_loader.dataset.dataset[data_loader.dataset.indices]

    correct = (np.array(predictions) == targets.numpy())
    label_0 = (np.array(predictions) == 0)
    label_1 = (np.array(predictions) == 1)
    verified = (np.array(ub_predictions) == np.array(lb_predictions)) & correct

    fig, ax = plt.subplots(figsize=(12, 12))
    plt.scatter(data[label_0 & correct][:, 0], data[label_0 & correct][:, 1], c="blue", label="0")
    plt.scatter(data[label_1 & correct][:, 0], data[label_1 & correct][:, 1], c="green", label="1")
    plt.scatter(data[label_0 & np.invert(correct)][:, 0], data[label_0 & np.invert(correct)][:, 1], c="cyan",
                label="0_incorrect")
    plt.scatter(data[label_1 & np.invert(correct)][:, 0], data[label_1 & np.invert(correct)][:, 1], c="lawngreen",
                label="1_incorrect")

    for v, c in zip([True, False], ["grey", "yellow"]):
        points = data[verified ^ (not v)]
        for i in range(len(points)):
            cir = plt.Circle((points[:, 0][i], points[:, 1][i]), radius=epsilon, color=c, fill=True, alpha=0.05)
            ax.add_patch(cir)

    ax.set_aspect('equal')
    plt.legend(loc='upper right')
    wandb.log({log_name: [wandb.Image(plt, caption="Predictions")]})
    plt.show()


def compute_accs(model, data_loader, threshold, ptb):
    """ Compute accuracy and verified accuracy
    
    model: base model
    data_loader: where to load data from
    threshold: assign class 0 if probability for class 0 is >= threshold
    ptb: perturbation norm
    """

    softmax = nn.Softmax(dim=1)

    acc = 0
    verified_acc = 0

    lb_classes, ub_classes, predicted_classes = [], [], []

    for data, target in tqdm(data_loader):
        model = BoundedModule(model, data)
        my_input = BoundedTensor(data, ptb)
        prediction = model(my_input)
        lb, ub = model.compute_bounds(x=(my_input,), method="backward")

        pred_class = (softmax(prediction).squeeze()[:, 0] < threshold).int()
        lb_class = (softmax(lb).squeeze()[:, 0] < threshold).int()
        ub_class = (softmax(ub).squeeze()[:, 0] < threshold).int()
        acc += torch.sum(pred_class == target)
        verified_acc += torch.sum(torch.logical_and(torch.logical_and(pred_class == target, lb_class == target),
                                                    ub_class == target))
        lb_classes += list(lb_class.detach().numpy())
        ub_classes += list(ub_class.detach().numpy())
        predicted_classes += list(pred_class.detach().numpy())

    acc = acc.item() / len(data_loader.dataset)
    verified_acc = verified_acc.item() / len(data_loader.dataset)

    return {
        "accuracy": acc,
        "verified_accuracy": verified_acc,
        "ub_classes": ub_classes,
        "lb_classes": lb_classes,
        "predicted_classes": predicted_classes
    }


def train_model(model, criterion, device, train_loader):
    config = wandb.config

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    pbar = tqdm(range(config.epochs))
    training_accuracy, training_loss = 0, 0
    for epoch in pbar:
        pbar.set_description(
            "(Loss: {:.6f}, Accuracy: {:.4f}) - Progress: ".format(training_loss, training_accuracy))
        training_accuracy, training_loss = train(model, device, train_loader, criterion, optimizer,
                                                 epoch)
        wandb.log({
            "epoch": epoch,
            "train_acc": training_accuracy,
            "train_loss": training_loss
        })
    torch.save(model, "model.torch")


def main():
    wandb.init(project='dependable-classification', entity='implication-elimination')
    config = wandb.config

    data = pd.read_excel(config.data_path)
    train_dataset, test_dataset = prepare_data(data)

    # Create simple network
    neurons = config.neurons
    model = nn.Sequential(
        nn.Linear(2, neurons),
        nn.ReLU(),
        nn.Linear(neurons, 2),
    )

    wandb.watch(model)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size)
    model_path = "model.torch"
    device = torch.device("cpu")
    criterion = nn.CrossEntropyLoss().to(device)

    # Train network / load already trained network
    if config.cache_model:
        try:
            model = torch.load(model_path)
        except FileNotFoundError:
            train_model(model, criterion, device, train_loader)
    else:
        train_model(model, criterion, device, train_loader)

    # Check test accuracy
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size)
    test_acc, test_loss = test(model, device, test_loader, criterion)
    wandb.log({"test_loss": test_loss})

    # Evaluate boundaries
    threshold = config.threshold
    ptb = PerturbationLpNorm(norm=np.inf, eps=config.eps)

    results_train = compute_accs(model, train_loader, threshold, ptb)
    train_acc = results_train["accuracy"]
    train_verified_acc = results_train["verified_accuracy"]
    plot_results(results_train["ub_classes"], results_train["lb_classes"], results_train["predicted_classes"],
                 train_loader, config.eps, "train_results")

    results_test = compute_accs(model, test_loader, threshold, ptb)
    plot_results(results_test["ub_classes"], results_test["lb_classes"], results_test["predicted_classes"],
                 test_loader, config.eps, "test_results")
    test_acc = results_test["accuracy"]
    test_verified_acc = results_test["verified_accuracy"]

    wandb.log({
        "train_acc": train_acc,
        "train_verified_acc": train_verified_acc,
        "test_acc": test_acc,
        "test_verified_acc": test_verified_acc
    })


if __name__ == '__main__':
    main()
