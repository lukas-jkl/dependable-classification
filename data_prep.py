import pandas as pd
import torch
from torch.utils.data import Dataset
import wandb

from typing import Tuple


def load_data(dataset_name: str, train: bool) -> pd.DataFrame:
    if train:
        data_path = f"./data/trainingdata_{dataset_name}.xls"
        data = pd.read_excel(data_path)
    else:
        data_path = f"./data/validation data sets/{dataset_name.capitalize()}.csv"
        data = pd.read_csv(data_path, sep=";")
    return data


def prepare_data(dataset_name: str, val_split: float, train: bool = True) -> Tuple[Dataset, Dataset]:
    """ Prepare data for training, split into datasets

    dataset_name: which dataset to load (a, b, c)
    val_split: the fraction of data used as validation data

    Returns:
        (Training set, Validation set)
    """
    data = load_data(dataset_name, train)

    X = torch.tensor(data.loc[:, ["x_i1", "x_i2"]].values).float()
    Y = torch.tensor(data.l_i.values)
    dataset = torch.utils.data.TensorDataset(X, Y)

    train_size = round(X.shape[0] * (1 - val_split))
    val_size = X.shape[0] - train_size

    if train:
        return torch.utils.data.random_split(dataset, (train_size, val_size),
                                             generator=torch.Generator().manual_seed(42))
    else:
        return torch.utils.data.Subset(dataset, list(range(len(dataset)))), None


def save_predictions(dataset_name: str, predictions: torch.Tensor, train: bool = True):
    data = load_data(dataset_name, train)
    data["l_i"] = predictions.squeeze()
    data = data.set_index(data.columns[0])
    data.index.name = ""

    file = f"artifacts/{dataset_name.capitalize()}.csv"
    data.to_csv(file, sep=";")
    artifact = wandb.Artifact('predictions', type='result')
    artifact.add_file(file)
    wandb.log_artifact(artifact)
