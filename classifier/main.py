import os
import sys
import random
import os.path as osp
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, StepLR

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

import pandas as pd

from adabound import AdaBound
from adabelief_pytorch import AdaBelief
from lion_pytorch import Lion
from adopt import ADOPT
from adan_pytorch import Adan

from model import Net, MNISTClassifier
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from adashift.optimizers import AdaShift
from PIDAO_SI import PIDAccOptimizer_SI
from pytorch_optimizers import AdamPlus, Lamb, Adam2, Adam3, Adam4
from pytorch_utils_opt import GradSNR


parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', '-lr', type=float, default=0.01, help='Learning rate')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--optimizer', '-o', default='Adam', help='Optimizer')
parser.add_argument('--seed', type=int, default=100, help='Random seed')
parser.add_argument('--beta_1', type=float, default=0.9, help='Coefficient for EMA of Adam+')
parser.add_argument('--beta_2', type=float, default=0.999, help='Coefficient for EMA of Adam+')
parser.add_argument('--db', type=float, default=-3, help='dB threshold in Adam+')
parser.add_argument('--decay_rate', type=float, default=1.0, help='Decay rate')
parser.add_argument('--decay_steps', type=int, default=1, help='Decay steps')
parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4, help='Weight decay')
parser.add_argument('--batch_size', type=int, default=32, help='LR scheduler type')
parser.add_argument('--model', type=str, default='resnet-18', help='Model')
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if __name__ == '__main__':
    dataset = "mnist"
    # model_str = "resnet-18"
    model_str = args.model
    batch_size = args.batch_size
    if 'cifar' in dataset:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),  # CIFAR-10 mean
                                 (0.2023, 0.1994, 0.2010))  # CIFAR-10 std
        ])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ])        
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform_test)
    else:
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.1307,), (0.3081,))
             ]
            )
        trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                                download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root='./data', train=False,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False)
    if model_str == "mnist":
        # model = Net()
        model = MNISTClassifier()
    if 'resnet' in model_str:
        if model_str == "resnet-18":
            model = models.resnet18(weights=None)
        elif model_str == "resnet-34":
            model = models.resnet34(weights=None)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, 10)

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    # K = 50
    # beta = 1-1/K
    # grad_snr_obj = GradSNR(K, beta)
    wd = args.weight_decay

    if args.optimizer.lower() == "Adam".lower():
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(args.beta_1, args.beta_2), weight_decay=wd)
    elif args.optimizer.lower() == "AdamW".lower():
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(args.beta_1, args.beta_2), weight_decay=wd)
    elif args.optimizer.lower() == "SGD".lower():
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=wd)
    elif args.optimizer.lower() == "AMSGrad".lower():
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(args.beta_1, args.beta_2), amsgrad=True, weight_decay=wd)
    elif args.optimizer.lower() == "AdaBound".lower():
        optimizer = AdaBound(model.parameters(), lr=args.learning_rate, betas=(args.beta_1, args.beta_2), weight_decay=wd)
    elif args.optimizer.lower() == "AdaBelief".lower():
        optimizer = AdaBelief(
            model.parameters(), lr=args.learning_rate, betas=(args.beta_1, args.beta_2),
            weight_decouple=False, weight_decay=wd,
            rectify=False
        )
    elif args.optimizer.lower() == "AdaBeliefW".lower():
        optimizer = AdaBelief(
            model.parameters(), lr=args.learning_rate, betas=(args.beta_1, args.beta_2),
            weight_decouple=True, weight_decay=wd,
            rectify=False
        )
    elif args.optimizer.lower() == "Adagrad".lower():
        optimizer = torch.optim.Adagrad(model.parameters(), lr=args.learning_rate, weight_decay=wd)
    elif args.optimizer.lower() == 'RMSprop'.lower():
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.learning_rate, alpha=args.beta2, weight_decay=wd)
    elif args.optimizer.lower() == "PIDAOSI".lower():
        optimizer = PIDAccOptimizer_SI(model.parameters(), lr=args.learning_rate, momentum=100/9, kp=1000/9, ki=0.1, kd=1)
    elif args.optimizer.lower() == "AdaShift".lower():
        optimizer = AdaShift(model.parameters(), lr=args.learning_rate, betas=(0.9, args.beta_2), weight_decay=wd)
    elif args.optimizer.lower() == "Adan".lower():
        optimizer = Adan(model.parameters(), lr=args.learning_rate, betas=(0.98, 0.92, args.beta_2), weight_decay=wd)
    elif args.optimizer.lower() == "Lion".lower():
        optimizer = Lion(model.parameters(), lr=args.learning_rate, weight_decay=wd)
    elif args.optimizer.lower() == "ADOPT".lower():
        optimizer = ADOPT(model.parameters(), lr=args.learning_rate, betas=(args.beta_1, args.beta_2), weight_decay=wd)
    elif args.optimizer.lower() == "AdamW".lower():
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(args.beta_1, args.beta_2), weight_decay=wd)
    elif args.optimizer.lower() == "AdamWPlus".lower():
        optimizer = AdamPlus(
            model.parameters(), lr=args.learning_rate, beta1=args.beta_1, beta2=args.beta_2,
            weight_decay=wd, weight_decouple=True,
            is_mask=False, db_threshold=args.db,
            db_noise=-140
        )
    elif args.optimizer.lower() == "AdamPlus".lower():
        optimizer = AdamPlus(
            model.parameters(), lr=args.learning_rate, beta1=args.beta_1, beta2=args.beta_2,
            weight_decay=wd, weight_decouple=False,
            is_mask=False, db_threshold=args.db,
            db_noise=-140
        )
    elif args.optimizer.lower() == "AdamW+".lower():
        optimizer = AdamPlus(
            model.parameters(), lr=args.learning_rate, beta1=args.beta_1, beta2=args.beta_2,
            weight_decay=wd, weight_decouple=True,
            is_mask=False, db_threshold=args.db,
            db_noise=-140
        )
    elif args.optimizer.lower() == "AMSGradPlus".lower():
        optimizer = AdamPlus(
            model.parameters(), lr=args.learning_rate, beta1=args.beta_1, beta2=args.beta_2, amsgrad=True,
            weight_decay=wd, weight_decouple=False,
            is_mask=False, db_threshold=args.db,
            db_noise=-140
        )
    elif args.optimizer.lower() == "Adam2".lower():
        optimizer = Adam2(
            model.parameters(), lr=args.learning_rate, beta1=args.beta_1, beta2=args.beta_2,
            weight_decay=wd, weight_decouple=False,
            is_lamb=False, option=0,
            db_noise=-140
        )
    elif args.optimizer.lower() == "AdamW2".lower():
        optimizer = Adam2(
            model.parameters(), lr=args.learning_rate, beta1=args.beta_1, beta2=args.beta_2,
            weight_decay=wd, weight_decouple=True,
            is_lamb=False, option=0,
            db_noise=-140
        )
    elif args.optimizer.lower() == "Adam22".lower():
        optimizer = Adam2(
            model.parameters(), lr=args.learning_rate, beta1=args.beta_1, beta2=args.beta_2,
            weight_decay=wd, weight_decouple=False,
            is_lamb=False, option=2,
            db_noise=-140
        )
    elif args.optimizer.lower() == "Lamb".lower():
        optimizer = Lamb(
            model.parameters(), lr=args.learning_rate, betas=(args.beta_1, args.beta_2),
            weight_decay=wd, weight_decouple=False,
        )
    elif args.optimizer.lower() == "Lamb2".lower():
        optimizer = Adam2(
            model.parameters(), lr=args.learning_rate, beta2=args.beta_2, eps=1e-10,
            weight_decay=wd, weight_decouple=False,
            is_lamb=True, option=0,
            db_noise=-140
        )
    elif args.optimizer.lower() == "LambPlus".lower():
        optimizer = AdamPlus(
            model.parameters(), lr=args.learning_rate, beta1=args.beta_1, beta2=args.beta_2,
            weight_decay=wd, weight_decouple=False,
            is_mask=False, db_threshold=args.db,
            is_lamb=True,
            db_noise=-140
        )
    elif args.optimizer.lower() == "Lamb4".lower():
        optimizer = Adam4(
            model.parameters(), lr=args.learning_rate, beta2=args.beta_2,
            weight_decay=wd, weight_decouple=False, option=0, 
            is_lamb=True,
        )
    elif args.optimizer.lower() == "Lamb41".lower():
        optimizer = Adam4(
            model.parameters(), lr=args.learning_rate, beta2=args.beta_2,
            weight_decay=wd, weight_decouple=False, option=1, 
            is_lamb=True,
        )
    elif args.optimizer.lower() == "Lamb42".lower():
        optimizer = Adam4(
            model.parameters(), lr=args.learning_rate, beta2=args.beta_2,
            weight_decay=wd, weight_decouple=True, option=2, 
            is_lamb=True,
        )
    elif args.optimizer.lower() == "Adam3".lower():
        optimizer = Adam3(
            model.parameters(), lr=args.learning_rate, beta2=args.beta_2,
            weight_decay=wd, weight_decouple=False,
            is_sign=True, db_threshold=args.db,
            db_noise=-140
        )
    elif args.optimizer.lower() == "Adam4".lower():
        optimizer = Adam4(
            model.parameters(), lr=args.learning_rate, beta2=args.beta_2,
            weight_decay=wd, weight_decouple=False, option=0,
        )
    elif args.optimizer.lower() == "AdamW4".lower():
        optimizer = Adam4(
            model.parameters(), lr=args.learning_rate, beta2=args.beta_2,
            weight_decay=wd, weight_decouple=True, option=0,
        )
    elif args.optimizer.lower() == "Adam41".lower():
        optimizer = Adam4(
            model.parameters(), lr=args.learning_rate, beta2=args.beta_2,
            weight_decay=wd, weight_decouple=False, option=1,
        )
    elif args.optimizer.lower() == "AdamW41".lower():
        optimizer = Adam4(
            model.parameters(), lr=args.learning_rate, beta2=args.beta_2,
            weight_decay=wd, weight_decouple=True, option=1,
        )
    elif args.optimizer.lower() == "Adam42".lower():
        optimizer = Adam4(
            model.parameters(), lr=args.learning_rate, beta2=args.beta_2,
            weight_decay=wd, weight_decouple=False, option=2,
        )
    else:
        raise ValueError("unsupported optimizer {}".format(args.optimizer))

    decay_rate = args.decay_rate
    decay_steps = args.decay_steps
    # scheduler = ExponentialLR(optimizer, gamma=decay_rate)
    scheduler = StepLR(optimizer, step_size=decay_steps, gamma=decay_rate)
    params = optimizer.param_groups[0]['params']
    num_params = len(params)
    results = pd.DataFrame([], columns=["epoch", "mini-batch", "loss_tr", "loss_val", "acc_tr", "acc_val", "SNR", "EmaSNR", "AdamPlusSNR", "m", "v", "g", "pwr_noise", "runtime"])
    results_list = []
    csvname = "{}/output/{}_{}_{}_{}_seed{}_{}_{}_{}_{}_TEST.csv".format(
        osp.dirname(osp.realpath(__file__)),
        model_str,
        args.optimizer,
        "{:.1e}".format(args.learning_rate),
        args.beta_2,
        args.seed,
        decay_rate,
        decay_steps,
        args.batch_size,
        "{:.1e}".format(args.weight_decay)
    )
    total_tr, loss_train, acc_train = 0, 0, 0
    # time_start = time.time()
    # with torch.no_grad():
    #     for i, data_tr in enumerate(trainloader, 0):
    #         inputs_tr, labels_tr = data_tr
    #         inputs_tr, labels_tr = inputs_tr.to(device), labels_tr.to(device)
    #         outputs_tr = model(inputs_tr)
    #         loss_tr = criterion(outputs_tr, labels_tr)
    #         _, predicted_tr = torch.max(outputs_tr, 1)
    #         total_tr += labels_tr.shape[0]
    #         correct_tr = (labels_tr == predicted_tr).sum().item()
    #         loss_train += loss_tr.item()
    #         acc_train += correct_tr
    #     total_te, acc_test, loss_test = 0, 0, 0
    #     for data_te in testloader:
    #         images_te, labels_te = data_te
    #         images_te, labels_te = images_te.to(device), labels_te.to(device)
    #         outputs_te = model(images_te)
    #         loss_te = criterion(outputs_te, labels_te)
    #         _, predicted_te = torch.max(outputs_te, 1)
    #         total_te += labels_te.shape[0]
    #         correct_te = (predicted_te == labels_te).sum().item()
    #         loss_test += loss_te.item()
    #         acc_test += correct_te
    # avg_loss_tr, avg_loss_te = loss_train/len(trainloader), loss_test/len(testloader)
    # avg_acc_tr, avg_acc_te = acc_train/total_tr, acc_test/total_te
    # e_runtime = time.time() - time_start

    results = pd.DataFrame([])
    upd_mod = 100
    for epoch in range(1, args.epochs+1):  # loop over the dataset multiple times
        total_tr, loss_train, acc_train = 0, 0, 0
        # gradient = [0 for _ in range(num_params)]
        time_start = time.time()
        for i, data_tr in enumerate(trainloader, 0):
            inputs_tr, labels_tr = data_tr
            inputs_tr, labels_tr = inputs_tr.to(device), labels_tr.to(device)
            optimizer.zero_grad()
            outputs_tr = model(inputs_tr)
            loss_tr = criterion(outputs_tr, labels_tr)
            loss_tr.backward()
            # for param_idx in range(num_params):
            #     param = params[param_idx]
            #     gradient[param_idx] += param.grad.cpu().detach().numpy()
            optimizer.step()
            _, predicted_tr = torch.max(outputs_tr, 1)
            total_tr += labels_tr.shape[0]
            correct_tr = (labels_tr == predicted_tr).sum().item()
            loss_train += loss_tr.item()
            acc_train += correct_tr
            if (i+1) % upd_mod == 0 and i > 0:
                e_runtime = time.time() - time_start
                total_te, acc_test, loss_test = 0, 0, 0
                with torch.no_grad():
                    for data_te in testloader:
                        images_te, labels_te = data_te
                        images_te, labels_te = images_te.to(device), labels_te.to(device)
                        outputs_te = model(images_te)
                        loss_te = criterion(outputs_te, labels_te)
                        _, predicted_te = torch.max(outputs_te, 1)
                        total_te += labels_te.shape[0]
                        correct_te = (predicted_te == labels_te).sum().item()
                        loss_test += loss_te.item()
                        acc_test += correct_te
                avg_loss_tr, avg_loss_te = loss_train/upd_mod, loss_test/len(testloader)
                avg_acc_tr, avg_acc_te = acc_train/total_tr, acc_test/total_te
                # g_norms = grad_snr_obj.add_sample(gradient)
                # elem_snrs_smv, snrs, noises = grad_snr_obj.compute_snr_smv()
                # elem_snrs_emv, snrs_ema, noises_ema = grad_snr_obj.compute_snr_emv()
                # snr_adam_plus = 0
                # if args.optimizer == 'AdamPlus':
                    # snr_adam_plus = optimizer.snr_layer()
                # m_norms, v_norms = [], []
                # m_norm, v_norm = 0, 0
                # g_norm = g_norms[-1]
                # if args.optimizer in ['Adam', 'AdamPlus']:
                #     for param in model.parameters():
                #         if param.grad is None:
                #             continue
                #         state = optimizer.state[param]
                #         m = state['exp_avg'].cpu().detach().numpy()
                #         v = state['exp_avg_sq'].cpu().detach().numpy()
                #         if m.ndim > 1:  # Filter bias
                #             m_norm = np.linalg.norm(m)
                #             v_norm = np.linalg.norm(v)
                #             m_norms.append(m_norm)
                #             v_norms.append(v_norm)
                #     m_norm, v_norm = m_norms[-1], v_norms[-1]

                log_items = {
                    "epoch": epoch,
                    "mini-batch": i+1,
                    "loss_tr": avg_loss_tr,
                    "loss_val": avg_loss_te,
                    "acc_tr": avg_acc_tr,
                    "acc_val": avg_acc_te,
                    "SNR": None, #snrs[-1],
                    "EmaSNR": None, #snrs_ema[-1],
                    "AdamPlusSNR": None, # snr_adam_plus,
                    "m": None, # m_norm,
                    "v": None, # v_norm,
                    "g": None, # g_norm,
                    "pwr_noise": None, # noises[-1],
                    "runtime": e_runtime,
                    "lr": scheduler.get_last_lr()[0]
                }
                print(' '.join(f"{k}: {v}" for k, v in log_items.items()))
                df_items = pd.DataFrame([log_items])
                results = pd.concat([results, df_items], ignore_index=True)
                results = results.reset_index(drop=True)
                results.to_csv(csvname)
                total_tr, loss_train, acc_train = 0, 0, 0
                gradient = [0 for _ in range(num_params)]
                time_start = time.time()

        if scheduler.get_last_lr()[0] > 0.000001:
            scheduler.step()
