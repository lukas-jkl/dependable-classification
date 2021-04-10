import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support


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


def prepare_data(data: pd.DataFrame, train_split: float, val_split: float) -> (Dataset, Dataset, Dataset):
    X = torch.tensor(data.loc[:, ["x_i1", "x_i2"]].values).float()
    Y = torch.tensor(data.l_i.values)
    dataset = torch.utils.data.TensorDataset(X, Y)
    train_size = round(X.shape[0] * train_split * (1 - val_split))
    val_size = round(X.shape[0] * train_split * val_split)
    test_size = X.shape[0] - train_size - val_size
    train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(
        dataset, (train_size, test_size, val_size)
    )
    return train_dataset, test_dataset, val_dataset


def train(model: nn.Sequential, device: torch.device, train_loader: DataLoader, criterion, optimizer) -> (float, float):
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


def plot_results(ub_predictions, lb_predictions, predictions, data_loader, pert_norm, epsilon, log_name):
    data, targets = data_loader.dataset.dataset[data_loader.dataset.indices]

    correct = (np.array(predictions) == targets.numpy())
    label_0 = (np.array(predictions) == 0)
    label_1 = (np.array(predictions) == 1)
    verified = (np.array(ub_predictions) == np.array(lb_predictions)) & correct

    fig, ax = plt.subplots(figsize=(12, 12))
    plt.scatter(data[label_0 & correct][:, 0], data[label_0 & correct][:, 1], c="mediumblue", label="0", s=10, zorder=10)
    plt.scatter(data[label_1 & correct][:, 0], data[label_1 & correct][:, 1], c="green", label="1", s=10, zorder=10)
    plt.scatter(data[label_0 & np.invert(correct)][:, 0], data[label_0 & np.invert(correct)][:, 1], c="midnightblue",
                label="0 (true: 1)", s=20, marker='x', zorder=15)
    plt.scatter(data[label_1 & np.invert(correct)][:, 0], data[label_1 & np.invert(correct)][:, 1], c="limegreen",
                label="1 (true: 0)", s=20, marker='x', zorder=15)

    for v, c in zip([True, False], ["grey", "yellow"]):
        points = data[verified ^ (not v)]
        for i in range(len(points)):
            if pert_norm == 2:
                patch = plt.Circle((points[:, 0][i], points[:, 1][i]), radius=epsilon, color=c, fill=True, alpha=0.05, zorder=0)
            elif pert_norm == np.inf:
                patch = plt.Rectangle((points[:, 0][i] - epsilon/2, points[:, 1][i] - epsilon/2), height=epsilon, width=epsilon, color=c, fill=True, alpha=0.05, zorder=0)
            else:
                raise RuntimeError(f"{pert_norm}-norm not supported for plotting")
            ax.add_patch(patch)

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


def train_model(model, criterion, device, train_loader, val_loader, num_epochs_early_stopping=0):
    config = wandb.config

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    pbar = tqdm(range(config.epochs))
    train_accuracy, train_loss = 0, 0
    val_accuracy, val_loss = 0, 0
    min_val_loss = None
    num_epochs_no_improvement = 0
    for epoch in pbar:
        pbar.set_description(
            "(Loss: {:.6f}, Accuracy: {:.4f}, Val_Loss: {:.6f}, Val_Accuracy: {:.4f}) - Progress: ".format(
                train_loss, train_accuracy, val_loss, val_accuracy))
        train_accuracy, train_loss = train(model, device, train_loader, criterion, optimizer)

        if len(val_loader.dataset) > 0:
            val_accuracy, val_loss = test(model, device, val_loader, criterion)
            if num_epochs_early_stopping:
                if min_val_loss is None or val_loss < min_val_loss:
                    min_val_loss = val_loss
                    num_epochs_no_improvement = 0
                else:
                    num_epochs_no_improvement += 1
                if num_epochs_no_improvement >= num_epochs_early_stopping:
                    print("Stopping training due to early stopping")
                    break

        wandb.log({
            "epoch": epoch,
            "train_acc": train_accuracy,
            "train_loss": train_loss,
            "val_acc": val_accuracy,
            "val_loss": val_loss
        })
    torch.save(model, "model.torch")


def precision_recall_f1_table(y_true, y_pred, labels=[0, 1]) -> pd.DataFrame:
    table_frame = pd.DataFrame(columns=["label", "precision", "recall", "f1", "support"])  # force column order
    precisions, recalls, fbetas, supports = precision_recall_fscore_support(y_true, y_pred, labels=labels)
    for label, prec, recall, fbeta, support in zip(labels, precisions, recalls, fbetas, supports):
        table_frame = table_frame.append(
            {"label": int(label), "precision": prec, "recall": recall, "f1": fbeta, "support": support},
            ignore_index=True)
    return table_frame


def main():
    wandb.init(project='dependable-classification', entity='implication-elimination', config='config.yaml')
    config = wandb.config
    config.model = 'NN'

    data_path = f"./data/trainingdata_{config.dataset}.xls"
    data = pd.read_excel(data_path)
    train_dataset, test_dataset, val_dataset = prepare_data(data, config.train_split, config.val_split)

    # Create simple network
    neurons = config.neurons
    model = nn.Sequential(
        nn.Linear(2, neurons),
        nn.ReLU(),
        nn.Linear(neurons, 2),
    )

    wandb.watch(model)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.batch_size)
    model_path = "model.torch"
    device = torch.device("cpu")
    criterion = nn.CrossEntropyLoss().to(device)

    # Train network / load already trained network
    if config.cache_model:
        try:
            model = torch.load(model_path)
        except FileNotFoundError:
            train_model(model, criterion, device, train_loader, val_loader, config.early_stopp_epochs)
    else:
        train_model(model, criterion, device, train_loader, val_loader, config.early_stopp_epochs)

    # Check test accuracy
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size)
    test_acc, test_loss = test(model, device, test_loader, criterion)
    wandb.log({"test_loss": test_loss})

    # Evaluate boundaries
    threshold = config.threshold
    ptb = PerturbationLpNorm(norm=config.pert_norm, eps=config.pert_eps)

    results_train = compute_accs(model, train_loader, threshold, ptb)
    train_acc = results_train["accuracy"]
    train_verified_acc = results_train["verified_accuracy"]
    plot_results(results_train["ub_classes"], results_train["lb_classes"], results_train["predicted_classes"],
                 train_loader, config.pert_norm, config.pert_eps, "train_results")
    train_table = precision_recall_f1_table(train_loader.dataset.dataset[train_loader.dataset.indices][1],
                                            results_train["predicted_classes"])
    train_wandb_table = wandb.Table(dataframe=train_table)
    wandb.log({"train_prec_recall_f1": train_wandb_table})

    results_test = compute_accs(model, test_loader, threshold, ptb)
    plot_results(results_test["ub_classes"], results_test["lb_classes"], results_test["predicted_classes"],
                 test_loader, config.pert_norm, config.pert_eps, "test_results")
    test_table = precision_recall_f1_table(test_loader.dataset.dataset[test_loader.dataset.indices][1],
                                           results_test["predicted_classes"])
    test_wandb_table = wandb.Table(dataframe=test_table)
    wandb.log({"test_prec_recall_f1": test_wandb_table})
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
