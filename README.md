<h1 align='center'>
  Sequential Latent Variable Models for<br> 
  Few-Shot High-Dimensional Time-Series Forecasting<br>
  (ICLR 2023 Top-25%)<br>
  [<a href='https://openreview.net/forum?id=7C9aRX2nBf2'>OpenReview</a>, <a href='https://iclr.cc/virtual/2023/oral/12551'>Presentation</a>]
</h1>

<p align='center'>*Xiajun Jiang, *Ryan Missel, Zhiyuan Li, Linwei Wang (*equal contribution)</p>

<p align='center'><img src="https://github.com/john-x-jiang/meta_ssm/assets/32918812/339d11ca-4110-44fe-bc80-ead1da08e02e" alt="framework schematic")/></p>
<p align='center'>Figure 1: Sequential latent-variable models (SLVM) framework for forecasting <br>high-dimensional sequences, based on the underlying PGM.</p>

## Description
Modern applications increasingly require learning and forecasting latent dynamics from high-dimensional time-series. Compared to univariate time-series forecasting, this adds a new challenge of reasoning about the latent dynamics of an unobserved abstract state. Sequential latent variable models (LVMs) present an attractive solution, although existing works either struggle with long-term forecasting or have difficulty learning across diverse dynamics. In this paper, we first present a conceptual framework of sequential LVMs to unify existing works, contrast their fundamental limitations, and identify an intuitive solution to long-term forecasting for diverse dynamics via meta-learning. We then present the first framework of few-shot forecasting for high-dimensional time-series: instead of learning a single dynamic function, we leverage data of diverse dynamics and learn to adapt latent dynamic functions to few-shot support series. This is realized via Bayesian meta-learning underpinned by: 1) a latent dynamic function conditioned on knowledge derived from few-shot support series, and 2) a meta-model that learns to extract such dynamic-specific knowledge via feed-forward embedding of support set. We compared the presented framework with a comprehensive set of baseline models trained 1) globally on the large meta-training set with diverse dynamics, and 2) individually on single dynamics, both with and without fine-tuning to k-shot support series used by the meta-models. We demonstrated that the presented framework is agnostic to the latent dynamic function of choice and, at meta-test time, is able to forecast for new dynamics given variable-shot of support series.

## Citation
Please cite the following if you use the data or the model in your work:
```bibtex
@inproceedings{
jiang2023sequential,
title={Sequential Latent Variable Models for Few-Shot High-Dimensional Time-Series Forecasting},
author={Xiajun Jiang and Ryan Missel and Zhiyuan Li and Linwei Wang},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=7C9aRX2nBf2}
}
```

## Requirements

* Python >= 3.8
* PyTorch >= 1.7
* ipdb
* matplotlib
* numpy
* scipy
* Pillow
* pymunk
* pygame
* torchdiffeq


## All Datasets
1. Bouncing ball dataset generated using [the code](http://www.cs.utoronto.ca/~ilya/code/2008/RTRBM.tar) provided with the original paper.
2. Mixed-physics datasets generated using [the code](https://github.com/deepmind/dm_hamiltonian_dynamics_suite) provided with the original paper
3. Check [here](https://drive.google.com/drive/folders/1Tm3DNrugcSbWXSNyeGL3jQKR8y3iXx0m?usp=share_link) for generated datasets.

## Cardiac Electrophysiology Model
1. Check [EP_model](https://github.com/john-x-jiang/meta_ssm/tree/main/EP_model) for the model specific to cardiac data.
