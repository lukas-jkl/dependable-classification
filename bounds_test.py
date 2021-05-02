import unittest
import torch
import wandb
import numpy as np
from tqdm import tqdm

import data_prep
import evaluation


class TestBounds(unittest.TestCase):
    """ Sanity check to check that the verrified regions do not interfere with each other

    """
    def test_bounds(self):
        wandb.init(mode='disabled', config='config.yaml')
        config = wandb.config

        assert config.cache_model_name is not None, "Cached model must exist to run this test"
        model = torch.load(config.cache_model_name).to("cpu")

        dataset, _ = data_prep.prepare_data(config.dataset, 0)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size)
        results = evaluation.evaluate_model_predictions(model, data_loader, config.pert_norm, config.pert_eps,
                                                        torch.device("cpu"))

        for i, (data, _) in enumerate(tqdm(dataset)):
            verified_label = results["verified_predicted_classes"][i]
            if verified_label != -1:
                for j, (compare_data, _) in enumerate(dataset):
                    compare_verified_label = results["verified_predicted_classes"][j]

                    if compare_verified_label != -1 and verified_label != compare_verified_label:
                        if config.pert_norm == np.inf:
                            ge = torch.all(compare_data >= data - config.pert_eps * 2)
                            le = torch.all(compare_data <= data + config.pert_eps * 2)
                            self.assertFalse(ge and le)
                        elif config.pert_norm == 2:
                            self.assertFalse(torch.sum((data - compare_data) ** 2) < config.pert_eps ** 2)


if __name__ == '__main__':
    unittest.main()
