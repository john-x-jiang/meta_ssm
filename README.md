# Sequential Latent Variable Models for Few-Shot High-Dimensional Time-Series Forecasting

This repository provides the code and data described in the paper:

[Sequential Latent Variable Models for Few-Shot High-Dimensional Time-Series Forecasting](https://openreview.net/forum?id=7C9aRX2nBf2)

\*Xiajun Jiang, \*Ryan Missel, Zhiyuan Li, Linwei Wang (\* equal contribution)

Published on ICLR 2023.

<!-- Please cite the following if you use the data or the model in your work:

```
@inproceedings{
jiang2023sequential,
title={Sequential Latent Variable Models for Few-Shot High-Dimensional Time-Series Forecasting},
author={Jiang, Xiajun and Missel, Ryan and Li, Zhiyuan and Wang, Linwei},
booktitle={Submitted to The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=7C9aRX2nBf2}
}
``` -->

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
