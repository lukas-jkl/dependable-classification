# Verified Classification using Deep Neural Networks with Perturbation Analysis

Here you can find the code for our submission to the *Siemens AI Dependability Assessment*. It solves a binary classification task using a deep neural network and provides security guarantees for local robustness using perturbation analysis based on linear relaxation. Our model displays high predictive performance, gives provable safety guarantees, scales well to more complex data sets, and lets domain experts dynamically configure the class-wise cost of misclassification.

## Authors:

- Patrick Deutschmann, patrick.deutschmann@student.tugraz.at
- Lukas Timpl, lukas.timpl@student.tugraz.at

## Submitted files:

- `config.yaml`: the [Weights & Biases](https://wandb.ai/site) config file for setting the training parameters
- `requirements.txt`: all used packages required to run our code
- `train.py`: the main script to train the model
- `nn_model.py` functions to create the model
- `evaluation.py`: evaluation functions
- `data_prep.py`: functions to prepare the data sets
- `bounds_test.py`: a sanity-check test case
- `baselines.ipynb`: notebook with the baseline computations
- `data_prep.ipynb`: notebook investigating the data sets

