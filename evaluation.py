from typing import Dict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
import torch
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from sklearn.metrics import precision_recall_fscore_support


def evaluate_model_predictions(model: nn.Module, data_loader: DataLoader, pert_norm, pert_eps: float,
                               device: torch.device) -> Dict:
    """ Compute predictions/accuracy as well as verified-predictions/verified-accuracy
         and costs associated with the predictions

    model: base model
    data_loader: where to load data from
    pert_norm: perturbation norm to use
    pert_eps: epsilon to use for perturbation norm
    device: the torch device on which to compute

    Returns:
        Dict: computed measures and predictions
    """

    if len(data_loader) == 0:
        return {}

    ptb = PerturbationLpNorm(norm=pert_norm, eps=pert_eps)

    acc = 0
    verified_acc = 0
    threshold = 0.5
    cost = 0

    verified_predicted_classes, predicted_classes = [], []

    for data, target in tqdm(data_loader):
        data, target = data.to(device), target.to(device)
        model = BoundedModule(model, data)
        my_input = BoundedTensor(data, ptb)
        prediction = model(my_input)
        lb, ub = model.compute_bounds(x=(my_input,), method="backward")

        pred_class = (torch.sigmoid(prediction) > threshold).int()
        lb_class = (torch.sigmoid(lb) > threshold).int()
        ub_class = (torch.sigmoid(ub) > threshold).int()
        acc += torch.sum(pred_class.squeeze() == target)

        wrong_pred = pred_class.squeeze() != target
        cost += torch.sum(wrong_pred & (target == 1)) * wandb.config.cost_1_classified_as_0
        cost += torch.sum(wrong_pred & (target == 0)) * wandb.config.cost_0_classified_as_1

        verified_prediction = pred_class.clone()
        verified_prediction[lb_class != ub_class] = -1

        verified_acc += torch.sum(verified_prediction.squeeze() == target)

        predicted_classes += list(pred_class.detach())
        verified_predicted_classes += list(verified_prediction.detach())

    acc = acc.item() / len(data_loader.dataset)
    verified_acc = verified_acc.item() / len(data_loader.dataset)

    return {
        "accuracy": acc,
        "verified_accuracy": verified_acc,
        "verified_predicted_classes": torch.stack(verified_predicted_classes).to("cpu"),
        "predicted_classes": torch.stack(predicted_classes).to("cpu"),
        "total_cost": cost.item(),
        "cost_per_sample": cost.item() / len(data_loader.dataset)
    }


def precision_recall_f1_table(y_true: torch.Tensor, y_pred: torch.Tensor, labels=[0, 1]) -> pd.DataFrame:
    """ Compute precision recall and f1 values

    y_true: the true predictions
    y_pred: the predictions of the model
    labels: the labels for which to compute the metrics. Default [0, 1]
    Returns:
        DataFrame with the results
    """

    table_frame = pd.DataFrame(columns=["label", "precision", "recall", "f1", "support"])  # force column order
    precisions, recalls, fbetas, supports = precision_recall_fscore_support(y_true, y_pred, labels=labels)
    for label, prec, recall, fbeta, support in zip(labels, precisions, recalls, fbetas, supports):
        table_frame = table_frame.append(
            {"label": int(label), "precision": prec, "recall": recall, "f1": fbeta, "support": support},
            ignore_index=True)
    return table_frame


def plot_results(verified_predictions: torch.Tensor, predictions: torch.Tensor, data_loader: DataLoader, pert_norm,
                 epsilon: float, log_name: str):
    """ Creates a plot of the given predictions

    verified_predictions: The predictions that are also verified (plotted with eps neighborhood)
    predictions: The predictions of the model
    data_loader: Data points to plot
    pert_norm: Which norm was used (2, np.inf)
    epsilon: The size of the eps region
    log_name: The name under which the plot is logged
    """

    data, targets = data_loader.dataset.dataset[data_loader.dataset.indices]

    correct = (predictions.squeeze() == targets)
    label_0 = (predictions == 0).squeeze()
    label_1 = (predictions == 1).squeeze()
    verified_label_0 = (verified_predictions == 0).squeeze()
    verified_label_1 = (verified_predictions == 1).squeeze()

    fig, ax = plt.subplots(figsize=(12, 12))
    plt.scatter(data[label_0 & correct][:, 0], data[label_0 & correct][:, 1], c="green", label="0", s=10, zorder=10)
    plt.scatter(data[label_1 & correct][:, 0], data[label_1 & correct][:, 1], c="red", label="1", s=10, zorder=10)
    plt.scatter(data[label_0 & np.invert(correct).bool()][:, 0], data[label_0 & np.invert(correct).bool()][:, 1],
                c="darkred", label="0 (true: 1)", s=20, marker='x', zorder=15)
    plt.scatter(data[label_1 & np.invert(correct).bool()][:, 0], data[label_1 & np.invert(correct).bool()][:, 1],
                c="limegreen", label="1 (true: 0)", s=20, marker='x', zorder=15)

    for idx, c in zip([verified_label_0, verified_label_1], ["green", "red"]):
        points = data[idx]
        for i in range(len(points)):
            if pert_norm == 2:
                patch = plt.Circle((points[:, 0][i], points[:, 1][i]), radius=epsilon, color=c, fill=True, alpha=0.1,
                                   zorder=0)
            elif pert_norm == np.inf:
                patch = plt.Rectangle((points[:, 0][i] - epsilon, points[:, 1][i] - epsilon), height=epsilon * 2,
                                      width=epsilon * 2, color=c, fill=True, alpha=0.1, zorder=0)
            else:
                raise RuntimeError(f"{pert_norm}-norm not supported for plotting")
            ax.add_patch(patch)

    ax.set_aspect('equal')
    plt.legend(loc='upper right')
    wandb.log({log_name: [wandb.Image(plt, caption="Predictions")]})
    plt.show()
