# Boosting Adam-like Optimizers with Signal-to-Noise Ratio Guided Updates

This repository contains the code from [Boosting Adam-like Optimizers with Signal-to-Noise Ratio Guided Updates](https://openreview.net/forum?id=5nN5nnRlnW&noteId=5nN5nnRlnW) paper. 

## Dependencies
See [requirements.txt](requirements.txt).

## Structure
[classifier](classifier) directory contains the code for [MNIST-MLP](classifier/main.py) and [CIFAR10-ResNet-18](classifier/main_kuangliu.py) experiments.

[gps](gps) directory contains the code for [ZINC-GT](gps/graph_gps.py) experiment.

[cleanrl/cleanrl](cleanrl/cleanrl) directory contains the code for [CartPole-DQN](cleanrl/cleanrl/dqn.py) and [MuJoCo-SAC](cleanrl/cleanrl/sac_continuous_action.py) experiments.

Each directory contains a run.ipynb and plot_metric.ipynb files. The first one sets up the simulation parameters and runs the experiments while the second one plots the results.
