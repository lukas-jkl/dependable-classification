# Ideas

## Key questions

- How to estimate upper bound for our model's uncertainty?

## Resources
*  Safety verification for deep neural networks with provable guarantees: https://secml2018.github.io/marta-secml2018.pdf

## Models

### Gaussian Processes (GPs)

* Great explanation for GPs in general: https://distill.pub/2019/visual-exploration-gaussian-processes/#FurtherReading
* Python example for classification: http://krasserm.github.io/2020/11/04/gaussian-processes-classification/
* Maybe easier to compute upper bounds? But expressive enough in high D?
    * Maybe helpful? https://arxiv.org/abs/1912.00071


### Bayesian Neural Networks (BNNs)

- Overview: https://www.kdnuggets.com/2017/12/what-bayesian-neural-network.html
- torchbnn: https://pypi.org/project/torchbnn/
  - Example: https://github.com/Harry24k/bayesian-neural-network-pytorch/blob/master/demos/Bayesian%20Neural%20Network%20Classification.ipynb
- Uncertainty in BNN: [What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?](https://arxiv.org/abs/1703.04977)



* Tutorials
  * https://towardsdatascience.com/making-your-neural-network-say-i-dont-know-bayesian-nns-using-pyro-and-pytorch-b1c24e6ab8cd
  * https://analyticsindiamag.com/hands-on-guide-to-bayesian-neural-network-in-classification/