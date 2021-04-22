import pandas as pd
import torch
from torch.utils.data import Dataset

from typing import Tuple

def prepare_data(dataset_name: str, val_split: float) -> Tuple[Dataset, Dataset]:
    data_path = f"./data/trainingdata_{dataset_name}.xls"
    data = pd.read_excel(data_path)

    X = torch.tensor(data.loc[:, ["x_i1", "x_i2"]].values).float()
    Y = torch.tensor(data.l_i.values)
    dataset = torch.utils.data.TensorDataset(X, Y)

    train_size = round(X.shape[0] * (1 - val_split))
    val_size = X.shape[0] - train_size

    return torch.utils.data.random_split(dataset, (train_size, val_size), generator=torch.Generator().manual_seed(42))