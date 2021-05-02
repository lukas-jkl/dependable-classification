# Verified Classification using Deep Neural Networks with Perturbation Analysis

Here you can find the code for our submission to the *Siemens AI Dependability Assessment*. It solves a binary classification task using a deep neural network and provides security guarantees for local robustness using perturbation analysis based on linear relaxation. Our model displays high predictive performance, gives provable safety guarantees, scales well to more complex data sets, and lets domain experts dynamically configure the class-wise cost of misclassification.

## Authors:

- Patrick Deutschmann
- Lukas Timpl

## Submitted files:

- `requirements.txt`: all used packages required to run our code
- `train.py`: the main script to train the model
- `nn_model.py` functions to create the model
- `evaluation.py`: evaluation functions
- `data_prep.py`: functions to prepare the datasets
- `bounds_test.py`: contains a sanity-check test case

