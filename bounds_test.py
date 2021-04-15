import unittest
import torch
import lirpa
import wandb
import pandas as pd
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
import numpy as np
from tqdm import tqdm


class TestBounds(unittest.TestCase):
    def test_bounds(self):
        wandb.init(mode='disabled', config='config.yaml')
        config = wandb.config
        model = torch.load("model.torch")

        threshold = config.threshold
        ptb = PerturbationLpNorm(norm=config.pert_norm, eps=config.pert_eps)

        data_path = f"./data/trainingdata_{config.dataset}.xls"
        data = pd.read_excel(data_path)
        dataset, _, _ = lirpa.prepare_data(data, 1, 0)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size)
        results = lirpa.compute_accs(model, data_loader, threshold, ptb)

        for i, (data, _) in enumerate(tqdm(dataset)):
            verified_label = results["verified_predicted_classes"][i]
            if verified_label != -1:
                for j, (compare_data, _) in enumerate(dataset):
                    compare_verified_label = results["verified_predicted_classes"][j]

                    if compare_verified_label != -1 and verified_label != compare_verified_label:
                        if config.pert_norm == np.inf:
                            ge = torch.all(compare_data >= data - config.pert_eps)
                            le = torch.all(compare_data <= data + config.pert_eps)
                            self.assertFalse(ge and le)
                        elif config.pert_norm == 2:
                            self.assertFalse(torch.sum((data - compare_data) ** 2) < config.pert_eps ** 2)


if __name__ == '__main__':
    unittest.main()
