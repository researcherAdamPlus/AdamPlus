# utils/optim.py
import sys
import os
import os.path as osp
import random

import torch
from adabound import AdaBound
from adabelief_pytorch import AdaBelief
from lion_pytorch import Lion
from adopt import ADOPT
from adan_pytorch import Adan
from adashift.optimizers import AdaShift
from PIDAO_SI import PIDAccOptimizer_SI
from pytorch_optimizers import AdamPlus, Adam2, Adam3, Lamb, Adam4, Adam5, AdamSNR
from pytorch_optimizers import ADOPTPlus, AdamPlusSNRlr, AdamPlusNL


def get_optimizer(params, lr, args):
    opt = args.optimizer.lower()
    db = args.db 

    if opt == "Adam".lower():
        optimizer = torch.optim.Adam(params, lr=lr, betas=(args.beta_1, args.beta_2), weight_decay=args.weight_decay)
    elif opt == "AdamSNR".lower():
        optimizer = AdamSNR(params, lr=lr, beta1=args.beta_1, beta2=args.beta_2, weight_decay=args.weight_decay, eps=1e-8)
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
        optimizer = PIDAccOptimizer_SI(params, lr=lr, momentum=100/27, kp=1000/27, ki=0.1, kd=1, weight_decay=args.weight_decay)
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
    elif opt == "AdamPlusSNRlr".lower() or opt == "Adam5".lower():
        optimizer = AdamPlusSNRlr(params, lr=lr, db_threshold=db, beta1=args.beta_1, beta2=args.beta_2, db_noise=args.db_noise, weight_decay=args.weight_decay, weight_decouple=False, is_lamb=False, is_mask=False)
    elif opt == "AdamPlusSNRlrNL1".lower() or opt == "Adam51".lower():
        optimizer = AdamPlusSNRlr(params, lr=lr, db_threshold=db, beta1=args.beta_1, beta2=args.beta_2, db_noise=args.db_noise, weight_decay=args.weight_decay, weight_decouple=False, is_lamb=False, is_mask=False, option=1)
    elif opt == "AdamPlusSNRlrNL2".lower() or opt == "Adam52".lower():
        optimizer = AdamPlusSNRlr(params, lr=lr, db_threshold=db, beta1=args.beta_1, beta2=args.beta_2, db_noise=args.db_noise, weight_decay=args.weight_decay, weight_decouple=False, is_lamb=False, is_mask=False, option=2)
    elif opt == "AdamPlusSNRlrNL3".lower() or opt == "Adam53".lower():
        optimizer = AdamPlusSNRlr(params, lr=lr, db_threshold=db, beta1=args.beta_1, beta2=args.beta_2, db_noise=args.db_noise, weight_decay=args.weight_decay, weight_decouple=False, is_lamb=False, is_mask=False, option=3)
    elif opt == "Adam+".lower():
        optimizer = AdamPlus(params, lr=lr, db_threshold=db, beta1=args.beta_1, beta2=args.beta_2, db_noise=args.db_noise, weight_decay=args.weight_decay, weight_decouple=False, is_lamb=False, is_mask=False)
    elif opt == "AMSGrad+".lower():
        optimizer = AdamPlus(params, lr=lr, db_threshold=db, beta1=args.beta_1, beta2=args.beta_2, db_noise=args.db_noise, weight_decay=args.weight_decay, weight_decouple=False, is_lamb=False, is_mask=False, amsgrad=True)
    elif opt == "AdamW+".lower():
        optimizer = AdamPlus(params, lr=lr, db_threshold=db, beta1=args.beta_1, beta2=args.beta_2, db_noise=args.db_noise, weight_decouple=True, weight_decay=args.weight_decay, is_lamb=False, is_mask=False)
    elif opt == "AdamPlus".lower():
        optimizer = AdamPlus(params, lr=lr, db_threshold=db, beta1=args.beta_1, beta2=args.beta_2, db_noise=args.db_noise, weight_decay=args.weight_decay, weight_decouple=False, is_lamb=False, is_mask=True)
    elif opt == "AdamWPlus".lower():
        optimizer = AdamPlus(params, lr=lr, db_threshold=db, beta1=args.beta_1, beta2=args.beta_2, db_noise=args.db_noise, weight_decouple=True, weight_decay=args.weight_decay, is_lamb=False, is_mask=True)
    elif opt == "LambPlus".lower():
        optimizer = AdamPlus(params, lr=lr, db_threshold=db, beta1=args.beta_1, beta2=args.beta_2, db_noise=args.db_noise, weight_decay=args.weight_decay, weight_decouple=False, is_lamb=True)
    elif opt == "LambPlusNL1".lower() or opt == "Lamb4".lower():
        optimizer = AdamPlusNL(
            params, lr=lr, beta1=args.beta_1, beta2=args.beta_2,
            weight_decay=args.weight_decay, weight_decouple=False, option=0, 
            is_lamb=True,
        )
    elif opt == "LambPlusNL2".lower() or opt == "Lamb41".lower():
        optimizer = AdamPlusNL(
            params, lr=lr, beta1=args.beta_1, beta2=args.beta_2,
            weight_decay=args.weight_decay, weight_decouple=False, option=1, 
            is_lamb=True,
        )
    elif opt == "LambPlusNL3".lower() or opt == "Lamb42".lower():
        optimizer = AdamPlusNL(
            params, lr=lr, beta1=args.beta_1, beta2=args.beta_2,
            weight_decay=args.weight_decay, weight_decouple=True, option=2, 
            is_lamb=True,
        )
    elif opt == "ADOPTPlus".lower() or opt == "Adam2".lower():
        optimizer = ADOPTPlus(params, lr=lr, beta1=args.beta_1, beta2=args.beta_2, db_noise=args.db_noise, weight_decay=args.weight_decay, weight_decouple=False, is_lamb=False)
    elif opt == "AdamPlusNL1".lower() or opt == "Adam4".lower():
        optimizer = AdamPlusNL(
            params, lr=lr, beta1=args.beta_1, beta2=args.beta_2,
            weight_decay=args.weight_decay, weight_decouple=False, option=0,
        )
    elif opt == "AdamWPlusNL1".lower() or opt == "AdamW4".lower():
        optimizer = AdamPlusNL(
            params, lr=lr, beta1=args.beta_1, beta2=args.beta_2,
            weight_decay=args.weight_decay, weight_decouple=True, option=0,
        )
    elif opt == "AdamPlusNL2".lower() or opt == "Adam41".lower():
        optimizer = AdamPlusNL(
            params, lr=lr, beta1=args.beta_1, beta2=args.beta_2,
            weight_decay=args.weight_decay, weight_decouple=False, option=1,
        )
    elif opt == "AdamWPlusNL2".lower() or opt == "AdamW41".lower():
        optimizer = AdamPlusNL(
            params, lr=lr, beta1=args.beta_1, beta2=args.beta_2,
            weight_decay=args.weight_decay, weight_decouple=True, option=1,
        )
    elif opt == "AdamPlusNL3".lower() or opt == "Adam42".lower():
        optimizer = AdamPlusNL(
            params, lr=lr, beta1=args.beta_1, beta2=args.beta_2,
            weight_decay=args.weight_decay, weight_decouple=False, option=2,
        )
    elif opt == "AdamWPlusNL3".lower() or opt == "AdamW42".lower():
        optimizer = AdamPlusNL(
            params, lr=lr, beta1=args.beta_1, beta2=args.beta_2,
            weight_decay=args.weight_decay, weight_decouple=True, option=2,
        )
    elif opt == "ADOPTWPlus".lower() or opt == "AdamW2".lower():
        optimizer = ADOPTPlus(params, lr=lr, beta1=args.beta_1, beta2=args.beta_2, db_noise=args.db_noise, weight_decay=args.weight_decay, weight_decouple=True, is_lamb=False)
    elif opt == "ADOPTLambPlus".lower() or opt == "Lamb2".lower():
        optimizer = ADOPTPlus(
            params, lr=lr, beta1=args.beta_1, beta2=args.beta_2, eps=1e-10,
            weight_decay=args.weight_decay, weight_decouple=False,
            is_lamb=True, option=0,
            db_noise=-140
        )
    elif opt == "Adam3".lower():
        optimizer = Adam3(
            params, lr=args.learning_rate, beta1=args.beta_1, beta2=args.beta_2,
            weight_decay=args.weight_decay, weight_decouple=False,
            is_sign=True, db_threshold=args.db,
            db_noise=-140
        )
    elif opt == "AdamWPlusSNRlr".lower() or opt == "AdamW5".lower():
        optimizer = AdamPlusSNRlr(params, lr=lr, db_threshold=db, beta1=args.beta_1, beta2=args.beta_2, db_noise=args.db_noise, weight_decay=args.weight_decay, weight_decouple=True, is_lamb=False, is_mask=False)
    elif opt == "AdamWPlusSNRlrNL1".lower() or opt == "AdamW51".lower():
        optimizer = AdamPlusSNRlr(params, lr=lr, db_threshold=db, beta1=args.beta_1, beta2=args.beta_2, db_noise=args.db_noise, weight_decay=args.weight_decay, weight_decouple=True, is_lamb=False, is_mask=False, option=1)
    else:
        raise ValueError(f"Optimizer {args.optimizer} does not exist.")

    return optimizer 