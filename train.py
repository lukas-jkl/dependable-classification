import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import numpy as np
from typing import Tuple
from sklearn.metrics import precision_score

import nn_model, data_prep, evaluation


def eval(model: nn.Sequential, device: torch.device, data_loader: DataLoader, criterion) -> Tuple[float, float, float]:
    """ evaluate the model on the given set """
    loss, acc = 0, 0
    f1_s = []

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
            f1_s.append(precision_score(target.to("cpu"), pred_class.to("cpu"), zero_division=0))

    return (acc / len(data_loader.dataset)), loss / len(data_loader.dataset), np.mean(f1_s)


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
    val_accuracy, val_loss, val_f1 = 0, 0, 0
    min_val_loss = None
    num_epochs_no_improvement = 0
    for epoch in pbar:
        pbar.set_description(
            "(Loss: {:.6f}, Accuracy: {:.4f}, Val_Loss: {:.6f}, Val_Acc: {:.4f}, Val_F1: {:.4f}) - Progress: ".format(
                train_loss, train_accuracy, val_loss, val_accuracy, val_f1))
        train_accuracy, train_loss = train(model, device, train_loader, criterion, optimizer)

        if len(val_loader.dataset) > 0:
            val_accuracy, val_loss, val_f1 = eval(model, device, val_loader, criterion)
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
            "val_loss": val_loss,
            "val_f1": val_f1
        })
    torch.save(model, "model.torch")


def evaluate(model, train_loader, val_loader, device):
    config = wandb.config

    # Train results
    results_train = evaluation.compute_accs(model, train_loader, config.threshold, config.pert_norm, config.pert_eps, device)
    evaluation.plot_results(results_train["verified_predicted_classes"], results_train["predicted_classes"],
                 train_loader, config.pert_norm, config.pert_eps, "train_results")
    train_samples = train_loader.dataset.dataset[train_loader.dataset.indices][1]
    train_table = evaluation.precision_recall_f1_table(train_samples, results_train["predicted_classes"])

    # Validation results
    results_val = evaluation.compute_accs(model, val_loader, config.threshold, config.pert_norm, config.pert_eps, device)
    evaluation.plot_results(results_val["verified_predicted_classes"], results_val["predicted_classes"],
                 val_loader, config.pert_norm, config.pert_eps, "val_results")
    val_samples = val_loader.dataset.dataset[val_loader.dataset.indices][1]
    val_table = evaluation.precision_recall_f1_table(val_samples, results_val["predicted_classes"])

    # log it all
    wandb.log({
        "train_prec_recall_f1": wandb.Table(dataframe=train_table),
        "val_prec_recall_f1": wandb.Table(dataframe=val_table),
        "train_acc": results_train["accuracy"],
        "val_acc": results_val["accuracy"],
        "train_verified_acc": results_train["verified_accuracy"],
        "val_verified_acc": results_val["verified_accuracy"],
        "train_label_1_prec": train_table[train_table.label == 1].precision.values[0],
        "val_label_1_prec": val_table[val_table.label == 1].precision.values[0]
    })


def main():
    wandb.init(project='dependable-classification', entity='implication-elimination', config='config.yaml')
    config = wandb.config
    config.model = 'NN'

    train_dataset, val_dataset = data_prep.prepare_data(config.dataset, config.val_split)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.batch_size)

    model = nn_model.create_model(config.num_hidden_layers, config.hidden_dim, config.dropout)
    print(model)

    device = torch.device(config.device)
    num_pos = torch.sum(train_dataset[:][1])
    pos_weight = (len(train_dataset) - num_pos) / num_pos
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)

    # Train (or load cached)
    if config.cache_model_name:
        try:
            model = torch.load(config.cache_model_name)
        except FileNotFoundError:
            train_model(model, criterion, device, train_loader, val_loader, config.patience)
    else:
        train_model(model, criterion, device, train_loader, val_loader, config.patience)

    # Evaluate
    evaluate(model, train_loader, val_loader, device)


if __name__ == '__main__':
    main()
