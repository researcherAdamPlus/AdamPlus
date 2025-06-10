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

from model import Net, MNISTClassifier
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.pytorch_optims import get_optimizer
from utils.logger import CSVLoggerClassifier


parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', '-lr', type=float, default=0.01, help='Learning rate')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--optimizer', '-o', default='AdamSNR', help='Optimizer')
parser.add_argument('--seed', type=int, default=100, help='Random seed')
parser.add_argument('--beta_1', type=float, default=0.9, help='Coefficient for EMA of Adam+')
parser.add_argument('--beta_2', type=float, default=0.999, help='Coefficient for EMA of Adam+')
parser.add_argument('--db', type=float, default=-3, help='dB threshold in Adam+')
parser.add_argument('--decay_rate', type=float, default=1.0, help='Decay rate')
parser.add_argument('--decay_steps', type=int, default=1, help='Decay steps')
parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4, help='Weight decay')
parser.add_argument('--batch_size', type=int, default=32, help='LR scheduler type')
parser.add_argument('--model', type=str, default='mnist', help='Model')
parser.add_argument('--db_noise', type=float, default=-140.0, help='Noise in db')
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
                                                download=True, transform=transform_train)
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

    wd = args.weight_decay
    optimizer = get_optimizer(model.parameters(), args.learning_rate, args)
    decay_rate = args.decay_rate
    decay_steps = args.decay_steps
    # scheduler = ExponentialLR(optimizer, gamma=decay_rate)
    scheduler = StepLR(optimizer, step_size=decay_steps, gamma=decay_rate)
    params = optimizer.param_groups[0]['params']
    num_params = len(params)

    csvname = "{}/output/{}_{}_{}_{}_seed{}_{}_{}_{}_{}_{}.csv".format(
        osp.dirname(osp.realpath(__file__)),
        model_str,
        args.optimizer,
        "{:.1e}".format(args.learning_rate),
        args.beta_2,
        args.seed,
        decay_rate,
        decay_steps,
        args.batch_size,
        "{:.1e}".format(args.weight_decay),
        args.db
    )
    logger = CSVLoggerClassifier(csvname)
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

    upd_mod = 100
    start_time = time.time()
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
                ema_snr, v = None, None
                if hasattr(optimizer, 'snr_layer'):
                    ema_snr = optimizer.snr_layer()
                if hasattr(optimizer, 'v_layer'):
                    v = optimizer.v_layer()
                logger.log_episode(epoch, i+1, {
                    "loss_tr": avg_loss_tr,
                    "loss_val": avg_loss_te,
                    "acc_tr": avg_acc_tr,
                    "acc_val": avg_acc_te,
                    "EmaSNR": ema_snr,
                    "v": v,
                    "lr": optimizer.param_groups[0]["lr"],
                    "runtime": time.time() - start_time
                })
                total_tr, loss_train, acc_train = 0, 0, 0
        if scheduler.get_last_lr()[0] > 0.000001:
            scheduler.step()
