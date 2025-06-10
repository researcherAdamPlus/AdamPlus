'''Train CIFAR10 with PyTorch.
Original https://github.com/kuangliu/pytorch-cifar
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import sys
import time
import random
import argparse
import numpy as np
import pandas as pd

# Additional Optimizers
from adabound import AdaBound
from adabelief_pytorch import AdaBelief
from lion_pytorch import Lion
from adopt import ADOPT
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.pytorch_optims import get_optimizer


from models import *
# from utils import progress_bar
import getpass
if os.getuid() == 1008:
    os.environ["USER"] = "fallback_user"


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.01, help='Learning rate')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--optimizer', '-o', default='Adam', help='Optimizer')
parser.add_argument('--seed', type=int, default=100, help='Random seed')
parser.add_argument('--beta_1', type=float, default=0.9, help='Coefficient for EMA of Adam+')
parser.add_argument('--beta_2', type=float, default=0.999, help='Coefficient for EMA of Adam+')
parser.add_argument('--db', type=float, default=0, help='dB threshold in Adam+')
parser.add_argument('--decay_rate', type=float, default=1.0, help='Decay rate')
parser.add_argument('--decay_steps', type=int, default=1, help='Decay steps')
parser.add_argument('--weight_decay', '-wd', type=float, default=5e-4, help='Weight decay')
parser.add_argument('--batch_size', type=int, default=32, help='LR scheduler type')
parser.add_argument('--db_noise', type=float, default=-140.0, help='noise in db')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# ✅ Enable TF32 for performance on Ampere GPUs (e.g., 3060 Ti)
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True

# # (Optional) Precision control if using PyTorch 2.1+
# try:
#     torch.set_float32_matmul_precision('high')  # 'medium' is slightly faster
# except AttributeError:
#     pass  # For older PyTorch versions

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

dataset = "cifar10"
model_str = "resnet-18"
batch_size = args.batch_size
wd = args.weight_decay
decay_rate = args.decay_rate
decay_steps = args.decay_steps

results = pd.DataFrame([], columns=["epoch", "mini-batch", "loss_tr", "loss_val", "acc_tr", "acc_val", "SNR", "EmaSNR", "AdamPlusSNR", "m", "v", "g", "pwr_noise", "runtime"])
thread_str = "{}_{}_{}_{}_{}_seed{}_{}_{}_{}_rerun".format(
    dataset,
    model_str,
    args.optimizer,
    "{:.1e}".format(args.learning_rate),
    args.beta_2,
    args.seed,
    decay_rate,
    decay_steps,
    args.batch_size
)
csvname = "{}/output/{}.csv".format(
    os.path.dirname(os.path.realpath(__file__)),
    thread_str,
)

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
# net = SimpleDLA()
net = net.to(device)

# ✅ Compile before DataParallel
# net = torch.compile(net, backend="inductor", mode="default")

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/{}_ckpt.pth'.format(thread_str))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    results = pd.read_csv(csvname, index_col=0)

criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=args.lr,
#                       momentum=0.9, weight_decay=5e-4)

optimizer = get_optimizer(net.parameters(), args.learning_rate, args)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)


# Training
def train(epoch):
    # print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #              % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    return train_loss/len(trainloader), correct/total


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/{}_ckpt.pth'.format(thread_str))
        best_acc = acc

    return test_loss/len(testloader), correct/total




for epoch in range(start_epoch, args.epochs):
    time_start = time.time()
    loss_tr, acc_tr = train(epoch)
    loss_te, acc_te = test(epoch)
    scheduler.step()
    e_runtime = time.time() - time_start

    log_items = {
        "epoch": epoch+1,
        "mini-batch": 500,
        "loss_tr": loss_tr,
        "loss_val": loss_te,
        "acc_tr": acc_tr,
        "acc_val": acc_te,
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