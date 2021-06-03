import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import Optimizer
from tqdm import tqdm
import wandb
import numpy as np
from typing import Tuple
from sklearn.metrics import precision_score

import nn_model
import data_prep
import evaluation


def acc_loss_f1(model: nn.Module, device: torch.device, data_loader: DataLoader, imbalance_weight_1: torch.Tensor,
                pos_weight: torch.Tensor) -> Tuple[float, float, float]:
    """ Compute acc, loss and f1-score on the given model and data

    model: The model which to evaluate
    device: The device on which to execute
    data_loader: Data on which to evaluate
    imbalance_weight_1: Weight to counter imbalance due to different number of data points per label
                        value > 1 will increase the weight for data points with label 1
    pos_weight: Weight used to model cost associated with misclassifying one label over the other
                value > 1 will prioritize the loss for predicting 0 for the true label 1

    Returns:
        (acc, loss, f1)
    """

    loss, acc = 0, 0
    f1_s = []

    # switch to evaluation mode
    model.eval()
    with torch.no_grad():
        for i, (data, target) in enumerate(data_loader):
            data, target = data.to(device).view(data.size(0), -1), target.to(device)

            imbalance_weights = torch.where(target == 1, imbalance_weight_1.to(device), torch.tensor(1.).to(
                device)).unsqueeze(dim=1)
            criterion = nn.BCEWithLogitsLoss(weight=imbalance_weights, pos_weight=pos_weight).to(device)

            # compute the output
            output = model(data)

            # compute the classification error and loss
            pred_class = (torch.sigmoid(output) > 0.5).int()
            acc += torch.sum(pred_class.squeeze() == target)
            loss += len(data) * criterion(output, target.unsqueeze(dim=1).float()).item()
            f1_s.append(precision_score(target.to("cpu"), pred_class.to("cpu"), zero_division=0, average='macro'))

    return (acc / len(data_loader.dataset)), loss / len(data_loader.dataset), np.mean(f1_s)


def train(model: nn.Module, device: torch.device, train_loader: DataLoader, optimizer: Optimizer,
          imbalance_weight_1: torch.Tensor, pos_weight: torch.Tensor) -> Tuple[float, float, float]:
    """ Train the model on the given dataset for a single epoch and return acc, loss, f1

    model: The model to train
    device: The device on which to execute
    data_loader: Data to train
    optimizer: Optimizer used to train the model
    imbalance_weight_1: Weight to counter imbalance due to different number of data points per label
                        value > 1 will increase the weight for data points with label 1
    pos_weight: Weight used to model cost associated with misclassifying one label over the other
                value > 1 will prioritize the loss for predicting 0 for the true label 1

    Returns:
        (acc, loss, f1)
    """

    loss, acc = 0, 0
    f1_s = []
    model.train()
    model.to(device)

    for data, target in train_loader:
        data, target = data.to(device).view(data.size(0), -1), target.to(device)

        imbalance_weights = torch.where(target == 1, imbalance_weight_1.to(device), torch.tensor(1.).to(
            device)).unsqueeze(dim=1)
        criterion = nn.BCEWithLogitsLoss(weight=imbalance_weights, pos_weight=pos_weight).to(device)

        # compute the output
        output = model(data)

        # compute the classification error and loss
        loss = criterion(output, target.unsqueeze(dim=1).float())
        pred_class = (torch.sigmoid(output) > 0.5).int()
        acc += torch.sum(pred_class.squeeze() == target)
        loss += len(data) * loss.item()
        f1_s.append(precision_score(target.to("cpu"), pred_class.to("cpu"), zero_division=0, average='macro'))

        # compute the gradient
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return (acc / len(train_loader.dataset)), loss / len(train_loader.dataset), np.mean(f1_s)


def train_model(model: nn.Module, device: torch.device, train_loader: DataLoader, val_loader: DataLoader,
                patience: int = 0):
    """ Train the model on the given dataset

    model: The model to train
    train_loader: Data loader containing data for training
    val_loader: Data loader containing data for validation
    patience: Number of epochs with no improvement on val_loader data after which training will be stopped.
    """

    config = wandb.config

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    num_1 = torch.sum(train_loader.dataset[:][1])
    imbalance_weight_1 = (len(train_loader.dataset) - num_1) / num_1
    pos_weight = torch.tensor(wandb.config.cost_1_classified_as_0 / wandb.config.cost_0_classified_as_1)

    pbar = tqdm(range(config.epochs))
    train_accuracy, train_loss, train_f1 = 0, 0, 0
    val_accuracy, val_loss, val_f1 = 0, 0, 0
    min_val_loss = None
    num_epochs_no_improvement = 0
    for epoch in pbar:
        pbar.set_description(
            "(Loss: {:.6f}, Accuracy: {:.4f}, F1: {:.4f}, Val_Loss: {:.6f}, Val_Acc: {:.4f}, Val_F1: {:.4f}) - Progress: ".format(
                train_loss, train_accuracy, train_f1, val_loss, val_accuracy, val_f1))
        train_accuracy, train_loss, train_f1 = train(model, device, train_loader, optimizer, imbalance_weight_1,
                                                     pos_weight)

        if len(val_loader.dataset) > 0:
            val_accuracy, val_loss, val_f1 = acc_loss_f1(model, device, val_loader, imbalance_weight_1, pos_weight)
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
            "train_f1": train_f1,
            "val_acc": val_accuracy,
            "val_loss": val_loss,
            "val_f1": val_f1
        })

    model_path = "model.torch"
    torch.save(model.to("cpu"), model_path)
    artifact = wandb.Artifact('model', type='model')
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)


def evaluate(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: torch.device):
    """ Compute all evaluation metrics of the model, plot results and log them

    model: The trained model to evaluate
    train_loader: Data loader containing the training data
    val_loader: Data loader containing the validation data
    device: The device on which to execute
    """

    config = wandb.config
    model.to(device)
    results = {}

    # Train results
    results_train = evaluation.evaluate_model_predictions(model, train_loader, config.pert_norm, config.pert_eps,
                                                          device)
    evaluation.plot_results(results_train["verified_predicted_classes"], results_train["predicted_classes"],
                            train_loader, config.pert_norm, config.pert_eps, "train_results")
    train_samples = train_loader.dataset.dataset[train_loader.dataset.indices][1]
    train_table = evaluation.precision_recall_f1_table(train_samples, results_train["predicted_classes"])
    results.update({"train_" + k: v for k, v in results_train.items()})
    results.update({
        "train_prec_recall_f1": wandb.Table(dataframe=train_table),
    })

    # Validation results
    results_val = evaluation.evaluate_model_predictions(model, val_loader, config.pert_norm, config.pert_eps, device)
    if len(results_val) != 0:
        evaluation.plot_results(results_val["verified_predicted_classes"], results_val["predicted_classes"],
                                val_loader, config.pert_norm, config.pert_eps, "val_results")
        val_samples = val_loader.dataset.dataset[val_loader.dataset.indices][1]
        val_table = evaluation.precision_recall_f1_table(val_samples, results_val["predicted_classes"])
        results.update({"val_" + k: v for k, v in results_val.items()})
        results.update({
            "val_prec_recall_f1": wandb.Table(dataframe=val_table),
        })

    # Full results
    full_dataset = ConcatDataset([train_loader.dataset, val_loader.dataset])
    full_loader = DataLoader(full_dataset, batch_size=config.batch_size)
    results_full = evaluation.evaluate_model_predictions(model, full_loader, config.pert_norm, config.pert_eps, device)
    data_prep.save_predictions(config.dataset, results_full["predicted_classes"], config.dataset_type_train)
    results.update({"full_" + k: v for k, v in results_full.items()})

    wandb.log(results)


def main():
    wandb.init(project='dependable-classification', entity='implication-elimination', config='config.yaml')
    config = wandb.config
    config.model = 'NN'

    train_dataset, val_dataset = data_prep.prepare_data(config.dataset, config.val_split, config.dataset_type_train)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

    model = nn_model.create_model(config.num_hidden_layers, config.hidden_dim, config.dropout)
    print(model)

    device = torch.device(config.device)

    # Train (or load cached)
    if config.cache_model_name:
        try:
            model = torch.load(config.cache_model_name)
        except FileNotFoundError:
            train_model(model, device, train_loader, val_loader, config.patience)
    else:
        train_model(model, device, train_loader, val_loader, config.patience)

    # Evaluate
    evaluate(model, train_loader, val_loader, device)


if __name__ == '__main__':
    main()
