# utils/optim.py
import sys
import os
import os.path as osp
import random

import torch
# from adabound import AdaBound
# from adabelief_pytorch import AdaBelief
# from lion_pytorch import Lion
# from adopt import ADOPT
# from adan_pytorch import Adan
# from adashift.optimizers import AdaShift
# from PIDAO_SI import PIDAccOptimizer_SI
# from muon import Muon
from pytorch_optimizers import AdamPlus, Adam2, Lamb, Adam4, Adam5, Adam6, AdamPlusv2


def get_optimizer(params, lr, args):
    opt = args.optimizer.lower()
    db = args.db 

    if opt == "Adam".lower():
        optimizer = torch.optim.Adam(params, lr=lr, betas=(args.beta_1, args.beta_2), weight_decay=args.weight_decay)
    elif opt == "SGD".lower():
        optimizer = torch.optim.SGD(params, momentum=0.9, lr=lr, weight_decay=args.weight_decay)
    elif opt == 'AMSGrad'.lower():
        optimizer = torch.optim.Adam(params, lr=lr, betas=(args.beta_1, args.beta_2), weight_decay=args.weight_decay, amsgrad=True)
    elif opt == "AdaBound".lower():
        optimizer = AdaBound(params, lr=lr, betas=(args.beta_1, args.beta_2), weight_decay=args.weight_decay)
    elif opt == "AdaBelief".lower():
        optimizer = AdaBelief(params, lr=lr, betas=(args.beta_1, args.beta_2), weight_decay=args.weight_decay,
                              weight_decouple=False, rectify=False, degenerated_to_sgd=False, print_change_log=False)
    elif opt == "AdaBeliefW".lower():
        optimizer = AdaBelief(params, lr=lr, betas=(args.beta_1, args.beta_2), weight_decay=args.weight_decay,
                              weight_decouple=True, rectify=False, degenerated_to_sgd=False, print_change_log=False)
    elif opt == "Adagrad".lower():
        optimizer = torch.optim.Adagrad(params, lr=lr, weight_decay=args.weight_decay)
    elif opt == "ADOPT".lower():
        optimizer = ADOPT(params, lr=lr, betas=(args.beta_1, args.beta_2), weight_decay=args.weight_decay)
    elif opt == "Lion".lower():
        optimizer = Lion(params, lr=lr, betas=(args.beta_1, args.beta_2), weight_decay=args.weight_decay)
    elif opt == "PIDAOSI".lower():
        optimizer = PIDAccOptimizer_SI(params, lr=lr, momentum=100/9, kp=1000/9 * 20, ki=0.1, kd=1, weight_decay=args.weight_decay)
    elif opt == "AdaShift".lower():
        optimizer = AdaShift(params, lr=lr, betas=(args.beta_1, args.beta_2))
    elif opt == "RMSprop".lower():
        optimizer = torch.optim.RMSprop(params, lr=lr, alpha=args.beta_2, weight_decay=args.weight_decay)
    elif opt == "Adan".lower():
        optimizer = Adan(params, lr=lr, betas=(0.98, 0.92, args.beta_2), weight_decay=args.weight_decay)
    elif opt == "AdamW".lower():
        optimizer = torch.optim.AdamW(params, lr=lr, betas=(args.beta_1, args.beta_2), weight_decay=args.weight_decay)
    elif opt == "Lamb".lower():
        optimizer = Lamb(params, lr=lr,  betas=(args.beta_1, args.beta_2), weight_decouple=False, weight_decay=args.weight_decay)
    elif opt == "Adam6".lower():
        optimizer = Adam6(params, lr=lr, db_threshold=db, beta1=args.beta_1, beta2=args.beta_2, db_noise=args.db_noise, weight_decay=args.weight_decay, weight_decouple=False, is_lamb=False, is_mask=False)
    elif opt == "Adam61".lower():
        optimizer = Adam6(params, lr=lr, db_threshold=db, beta1=args.beta_1, beta2=args.beta_2, db_noise=args.db_noise, weight_decay=args.weight_decay, weight_decouple=False, is_lamb=False, is_mask=False, option=1)
    elif opt == "AdamC6".lower():
        optimizer = Adam6(params, lr=lr, db_threshold=db, beta1=args.beta_1, beta2=args.beta_2, db_noise=args.db_noise, weight_decay=args.weight_decay, weight_decouple=False, is_lamb=False, is_mask=True)
    elif opt == "AdamC61".lower():
        optimizer = Adam6(params, lr=lr, db_threshold=db, beta1=args.beta_1, beta2=args.beta_2, db_noise=args.db_noise, weight_decay=args.weight_decay, weight_decouple=False, is_lamb=False, is_mask=True, option=1)
    elif opt == "Adam5".lower():
        optimizer = Adam5(params, lr=lr, db_threshold=db, beta1=args.beta_1, beta2=args.beta_2, db_noise=args.db_noise, weight_decay=args.weight_decay, weight_decouple=False, is_lamb=False, is_mask=False)
    elif opt == "Adam51".lower():
        optimizer = Adam5(params, lr=lr, db_threshold=db, beta1=args.beta_1, beta2=args.beta_2, db_noise=args.db_noise, weight_decay=args.weight_decay, weight_decouple=False, is_lamb=False, is_mask=False, option=1)
    elif opt == "Adam52".lower():
        optimizer = Adam5(params, lr=lr, db_threshold=db, beta1=args.beta_1, beta2=args.beta_2, db_noise=args.db_noise, weight_decay=args.weight_decay, weight_decouple=False, is_lamb=False, is_mask=False, option=2)
    elif opt == "Adam53".lower():
        optimizer = Adam5(params, lr=lr, db_threshold=db, beta1=args.beta_1, beta2=args.beta_2, db_noise=args.db_noise, weight_decay=args.weight_decay, weight_decouple=False, is_lamb=False, is_mask=False, option=3)
    elif opt == "Adam54".lower():
        optimizer = Adam5(params, lr=lr, db_threshold=db, beta1=args.beta_1, beta2=args.beta_2, db_noise=args.db_noise, weight_decay=args.weight_decay, weight_decouple=False, is_lamb=False, is_mask=False, option=4)
    elif opt == "Adam55".lower():
        optimizer = Adam5(params, lr=lr, db_threshold=db, beta1=args.beta_1, beta2=args.beta_2, db_noise=args.db_noise, weight_decay=args.weight_decay, weight_decouple=False, is_lamb=False, is_mask=False, option=5)
    elif opt == "AdamC53".lower():
        optimizer = Adam5(params, lr=lr, db_threshold=db, beta1=args.beta_1, beta2=args.beta_2, db_noise=args.db_noise, weight_decay=args.weight_decay, weight_decouple=False, is_lamb=False, is_mask=True, option=3)
    elif opt == "AdamC54".lower():
        optimizer = Adam5(params, lr=lr, db_threshold=db, beta1=args.beta_1, beta2=args.beta_2, db_noise=args.db_noise, weight_decay=args.weight_decay, weight_decouple=False, is_lamb=False, is_mask=True, option=4)
    elif opt == "AdamC55".lower():
        optimizer = Adam5(params, lr=lr, db_threshold=db, beta1=args.beta_1, beta2=args.beta_2, db_noise=args.db_noise, weight_decay=args.weight_decay, weight_decouple=False, is_lamb=False, is_mask=True, option=5)
    elif opt == "Adam+".lower():
        optimizer = AdamPlus(params, lr=lr, db_threshold=db, beta1=args.beta_1, beta2=args.beta_2, db_noise=args.db_noise, weight_decay=args.weight_decay, weight_decouple=False, is_lamb=False, is_mask=False)
    elif opt == "Adam+v2".lower():
        optimizer = AdamPlusv2(params, lr=lr, db_threshold=db, beta1=args.beta_1, beta2=args.beta_2, db_noise=args.db_noise, weight_decay=args.weight_decay, weight_decouple=False, is_lamb=False, is_mask=False)
    elif opt == "AdamF+".lower():
        optimizer = AdamPlus(params, lr=lr, db_threshold=db, beta1=args.beta_1, beta2=args.beta_2, db_noise=args.db_noise, weight_decay=args.weight_decay, weight_decouple=False, is_lamb=False, is_mask=False, is_fir=True)
    elif opt == "AMSGrad+".lower():
        optimizer = AdamPlus(params, lr=lr, db_threshold=db, beta1=args.beta_1, beta2=args.beta_2, db_noise=args.db_noise, weight_decay=args.weight_decay, weight_decouple=False, is_lamb=False, is_mask=False, amsgrad=True)
    elif opt == "AdamW+".lower():
        optimizer = AdamPlus(params, lr=lr, db_threshold=db, beta1=args.beta_1, beta2=args.beta_2, db_noise=args.db_noise, weight_decouple=True, weight_decay=args.weight_decay, is_lamb=False, is_mask=False)
    elif opt == "AdamPlus".lower():
        optimizer = AdamPlus(params, lr=lr, db_threshold=db, beta1=args.beta_1, beta2=args.beta_2, db_noise=args.db_noise, weight_decay=args.weight_decay, weight_decouple=False, is_lamb=False, is_mask=True)
    elif opt == "AdamFPlus".lower():
        optimizer = AdamPlus(params, lr=lr, db_threshold=db, beta1=args.beta_1, beta2=args.beta_2, db_noise=args.db_noise, weight_decay=args.weight_decay, weight_decouple=False, is_lamb=False, is_mask=True, is_fir=True)
    elif opt == "AdamPlus2".lower():
        optimizer = AdamPlus(params, lr=lr, db_threshold=db, beta1=args.beta_1, beta2=args.beta_2, db_noise=args.db_noise, weight_decay=args.weight_decay, weight_decouple=False, is_lamb=False, is_mask=True, mask_opt=2)
    elif opt == "AdamWPlus".lower():
        optimizer = AdamPlus(params, lr=lr, db_threshold=db, beta1=args.beta_1, beta2=args.beta_2, db_noise=args.db_noise, weight_decouple=True, weight_decay=args.weight_decay, is_lamb=False, is_mask=True)
    elif opt == "AdamC".lower():
        optimizer = AdamPlus(params, lr=lr, db_threshold=db, beta1=0, beta2=args.beta_2, db_noise=args.db_noise, weight_decay=args.weight_decay, weight_decouple=False, is_lamb=False, is_mask=True)
    elif opt == "AdamFLM".lower():
        optimizer = AdamPlus(params, lr=lr, db_threshold=db, beta1=0.5, beta2=args.beta_2, db_noise=args.db_noise, weight_decay=args.weight_decay, weight_decouple=False, is_lamb=False, is_mask=True, is_fir=True)
    elif opt == "AdamFNM".lower():
        optimizer = AdamPlus(params, lr=lr, db_threshold=db, beta1=0, beta2=args.beta_2, db_noise=args.db_noise, weight_decay=args.weight_decay, weight_decouple=False, is_lamb=False, is_mask=True, is_fir=True)
    elif opt == "AdamWNM".lower():
        optimizer = AdamPlus(params, lr=lr, db_threshold=db, beta1=0, beta2=args.beta_2, db_noise=args.db_noise, weight_decouple=True, weight_decay=args.weight_decay, is_lamb=False, is_mask=True)
    elif opt == "LambPlus".lower():
        optimizer = AdamPlus(params, lr=lr, db_threshold=db, beta1=args.beta_1, beta2=args.beta_2, db_noise=args.db_noise, weight_decay=args.weight_decay, weight_decouple=False, is_lamb=True)
    elif opt == "Lamb4".lower():
        optimizer = Adam4(
            params, lr=lr, beta1=args.beta_1, beta2=args.beta_2,
            weight_decay=args.weight_decay, weight_decouple=False, option=0, 
            is_lamb=True,
        )
    elif opt == "Lamb41".lower():
        optimizer = Adam4(
            params, lr=lr, beta1=args.beta_1, beta2=args.beta_2,
            weight_decay=args.weight_decay, weight_decouple=False, option=1, 
            is_lamb=True,
        )
    elif opt == "Lamb42".lower():
        optimizer = Adam4(
            params, lr=lr, beta1=args.beta_1, beta2=args.beta_2,
            weight_decay=args.weight_decay, weight_decouple=True, option=2, 
            is_lamb=True,
        )
    elif args.optimizer == "Adam2":
        optimizer = Adam2(params, lr=lr, beta1=args.beta_1, beta2=args.beta_2, db_noise=args.db_noise, weight_decay=args.weight_decay, weight_decouple=False, is_lamb=False)
    elif opt == "Adam4".lower():
        optimizer = Adam4(
            params, lr=lr, beta1=args.beta_1, beta2=args.beta_2,
            weight_decay=args.weight_decay, weight_decouple=False, option=0,
        )
    elif opt == "AdamW4".lower():
        optimizer = Adam4(
            params, lr=lr, beta1=args.beta_1, beta2=args.beta_2,
            weight_decay=args.weight_decay, weight_decouple=True, option=0,
        )
    elif opt == "Adam41".lower():
        optimizer = Adam4(
            params, lr=lr, beta1=args.beta_1, beta2=args.beta_2,
            weight_decay=args.weight_decay, weight_decouple=False, option=1,
        )
    elif opt == "AdamW41".lower():
        optimizer = Adam4(
            params, lr=lr, beta1=args.beta_1, beta2=args.beta_2,
            weight_decay=args.weight_decay, weight_decouple=True, option=1,
        )
    elif opt == "Adam42".lower():
        optimizer = Adam4(
            params, lr=lr, beta1=args.beta_1, beta2=args.beta_2,
            weight_decay=args.weight_decay, weight_decouple=False, option=2,
        )
    elif opt == "AdamW42".lower():
        optimizer = Adam4(
            params, lr=lr, beta1=args.beta_1, beta2=args.beta_2,
            weight_decay=args.weight_decay, weight_decouple=True, option=2,
        )
    else:
        raise ValueError(f"Optimizer {args.optimizer} does not exist.")

    return optimizer 