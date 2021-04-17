import pandas as pd
import torch
from torch.utils.data import Dataset

from typing import Tuple

def prepare_data(dataset_name: str, train_split: float, val_split: float) -> Tuple[Dataset, Dataset, Dataset]:
    data_path = f"./data/trainingdata_{dataset_name}.xls"
    data = pd.read_excel(data_path)

    X = torch.tensor(data.loc[:, ["x_i1", "x_i2"]].values).float()
    Y = torch.tensor(data.l_i.values)
    dataset = torch.utils.data.TensorDataset(X, Y)

    train_size = round(X.shape[0] * train_split * (1 - val_split))
    val_size = round(X.shape[0] * train_split * val_split)
    test_size = X.shape[0] - train_size - val_size

    return torch.utils.data.random_split(dataset, (train_size, test_size, val_size))