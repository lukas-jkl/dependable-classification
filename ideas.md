# Ideas

## Key questions

- How to estimate upper bound for our model's uncertainty?

## Resources

*  Safety verification for deep neural networks with provable guarantees: https://secml2018.github.io/marta-secml2018.pdf

## Models

### Probabilistic safety for general NNs

* Reluplex (2017): https://arxiv.org/pdf/1702.01135.pdf
  * slow, exact method
* Towards Fast Computation of Certified Robustness for ReLU Networks (2018): http://proceedings.mlr.press/v80/weng18a/weng18a.pdf
* Efficient Neural Network Robustness Certification with General Activation Functions (2018): https://arxiv.org/pdf/1811.00866.pdf
* On the Effectiveness of Interval Bound Propagation for Training Verifiably Robust Models (2019): https://arxiv.org/abs/1810.12715
  * (nice) code by Deepmind: https://github.com/deepmind/interval-bound-propagation

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
- Variational inference: https://krasserm.github.io/2019/03/14/bayesian-neural-networks/ 



* Tutorials
  * https://towardsdatascience.com/making-your-neural-network-say-i-dont-know-bayesian-nns-using-pyro-and-pytorch-b1c24e6ab8cd
  * https://analyticsindiamag.com/hands-on-guide-to-bayesian-neural-network-in-classification/

* Statistical guarantees:
	* [Statistical Guarantees for the Robustness of Bayesian Neural Networks](https://arxiv.org/abs/1903.01980)
		* citing papers: https://scholar.google.com/scholar?cites=12927215106318070846&as_sdt=2005&sciodt=0,5&hl=en
	* [Uncertainty Quantification with Statistical Guarantees in End-to-End Autonomous Driving Control](https://ieeexplore.ieee.org/abstract/document/9196844?casa_token=yi_N16OwFSAAAAAA:tTle5B_cSIjaF9PbITlfYJJZDh1LeqRllXRjz7LPlrNVV8uQWSqHzT7_xOxEiD4mHJOD_tHGIDyKLA)