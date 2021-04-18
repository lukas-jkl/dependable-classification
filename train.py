import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from typing import Tuple

import nn_model, data_prep, evaluation


def test(model: nn.Sequential, device: torch.device, data_loader: DataLoader, criterion) -> Tuple[float, float]:
    """ evaluate the model on the given set """
    loss, acc = 0, 0

    # switch to evaluation mode
    model.eval()
    with torch.no_grad():
        for i, (data, target) in enumerate(data_loader):
            data, target = data.to(device).view(data.size(0), -1), target.to(device)

            # compute the output
            output = model(data)

            # compute the classification error and loss
            pred_class = (torch.sigmoid(output) > wandb.config.threshold).int()
            acc += torch.sum(pred_class.squeeze() == target)
            loss += len(data) * criterion(output, target.unsqueeze(dim=1).float()).item()

    return (acc / len(data_loader.dataset)), loss / len(data_loader.dataset)


def train(model: nn.Sequential, device: torch.device, train_loader: DataLoader, criterion, optimizer) -> Tuple[float, float]:
    loss, acc = 0, 0
    
    model.train()
    model.to(device)

    for data, target in train_loader:
        data, target = data.to(device).view(data.size(0), -1), target.to(device)

        # compute the output
        output = model(data)

        # compute the classification error and loss
        loss = criterion(output, target.unsqueeze(dim=1).float())
        pred_class = (torch.sigmoid(output) > wandb.config.threshold).int()
        acc += torch.sum(pred_class.squeeze() == target)
        loss += len(data) * loss.item()

        # compute the gradient
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return (acc / len(train_loader.dataset)), loss / len(train_loader.dataset)


def train_model(model, criterion, device, train_loader, val_loader, patience=0):
    config = wandb.config

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

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
            if patience:
                if min_val_loss is None or val_loss < min_val_loss:
                    min_val_loss = val_loss
                    num_epochs_no_improvement = 0
                else:
                    num_epochs_no_improvement += 1
                if num_epochs_no_improvement >= patience:
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


def evaluate(model, train_loader, test_loader):
    config = wandb.config

    # Train results
    results_train = evaluation.compute_accs(model, train_loader, config.threshold, config.pert_norm, config.pert_eps)
    evaluation.plot_results(results_train["verified_predicted_classes"], results_train["predicted_classes"],
                 train_loader, config.pert_norm, config.pert_eps, "train_results")
    train_samples = train_loader.dataset.dataset[train_loader.dataset.indices][1]
    train_table = evaluation.precision_recall_f1_table(train_samples, results_train["predicted_classes"])
    verified_train_table = evaluation.precision_recall_f1_table(train_samples, results_train["verified_predicted_classes"])

    # Test results
    results_test = evaluation.compute_accs(model, test_loader, config.threshold, config.pert_norm, config.pert_eps)
    evaluation.plot_results(results_test["verified_predicted_classes"], results_test["predicted_classes"],
                 test_loader, config.pert_norm, config.pert_eps, "test_results")
    test_samples = train_loader.dataset.dataset[test_loader.dataset.indices][1]
    test_table = evaluation.precision_recall_f1_table(test_samples, results_test["predicted_classes"])
    verified_test_table = evaluation.precision_recall_f1_table(test_samples, results_test["verified_predicted_classes"])

    # log it all
    wandb.log({
        "train_prec_recall_f1": wandb.Table(dataframe=train_table),
        "test_prec_recall_f1": wandb.Table(dataframe=test_table),
        "test_verified_prec_recall_f1": wandb.Table(dataframe=verified_test_table),
        "train_acc": results_train["accuracy"],
        "train_verified_acc": results_train["verified_accuracy"],
        "train_label_1_prec": train_table[train_table.label == 1].precision.values[0],
        "train_verified_label_1_prec": verified_train_table[verified_train_table.label == 1].precision.values[0],
        "test_acc": results_test["accuracy"],
        "test_verified_acc": results_test["verified_accuracy"],
        "test_verified_label_1_prec": verified_test_table[verified_test_table.label == 1].precision.values[0],
        "test_label_1_prec": test_table[test_table.label == 1].precision.values[0]
    })


def main():
    wandb.init(project='dependable-classification', entity='implication-elimination', config='config.yaml')
    config = wandb.config
    config.model = 'NN'

    train_dataset, test_dataset, val_dataset = data_prep.prepare_data(config.dataset, config.train_split, config.val_split)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.batch_size)

    model = nn_model.create_model(config.num_hidden_layers, config.hidden_dim)
    print(model)

    device = torch.device("cpu")
    criterion = nn.BCEWithLogitsLoss().to(device) # TODO maybe try weights

    # Train (or load cached)
    if config.cache_model_name:
        try:
            model = torch.load(config.cache_model_name)
        except FileNotFoundError:
            train_model(model, criterion, device, train_loader, val_loader, config.patience)
    else:
        train_model(model, criterion, device, train_loader, val_loader, config.patience)

    # Test
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size)
    _, test_loss = test(model, device, test_loader, criterion)
    wandb.log({"test_loss": test_loss})

    # Evaluate
    evaluate(model, train_loader, test_loader)


if __name__ == '__main__':
    main()
