# Boosting Adam-like Optimizers with Signal-to-Noise Ratio Guided Updates

This repository contains the code from Boosting Adam-like Optimizers with Signal-to-Noise Ratio Guided Updates paper. 

## Dependencies
See [requirements.txt](requirements.txt).

## Structure
[classifier](classifier) directory contains the code for [MNIST-MLP](classifier/main.py) and [CIFAR10-ResNet-18](classifier/main_kuangliu.py) experiments.

[gps](gps) directory contains the code for [ZINC-GT](gps/graph_gps.py) experiment.

[cleanrl/cleanrl](cleanrl/cleanrl) directory contains the code for [CartPole-DQN](cleanrl/cleanrl/dqn.py) and [MuJoCo-SAC](cleanrl/cleanrl/sac_continuous_action.py) experiments.

[LSTM](LSTM) directory is the LMTM experiment from AdaBelief

[cramming_bert](cramming_bert) is based on crammed BERT from the corresponding paper

[nanoGPT](nanogpt) is based on nanoGPT codebase from the original repository

[atari](atari) implements atari games. It is based on cleanrl repository 
