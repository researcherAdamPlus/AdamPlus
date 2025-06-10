import os
import sys
import random
import time
import argparse
from typing import Any, Dict, Optional
import pandas as pd

import torch
from torch.nn import (
    BatchNorm1d,
    Embedding,
    Linear,
    ModuleList,
    ReLU,
    Sequential,
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.pytorch_optims import get_optimizer


import torch_geometric.transforms as T
from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, GPSConv, global_add_pool
from torch_geometric.nn.attention import PerformerAttention

path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', 'ZINC-PE')
transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')
train_dataset = ZINC(path, subset=True, split='train', pre_transform=transform)
val_dataset = ZINC(path, subset=True, split='val', pre_transform=transform)
test_dataset = ZINC(path, subset=True, split='test', pre_transform=transform)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--attn_type', default='multihead',
    help="Global attention type such as 'multihead' or 'performer'.")
parser.add_argument('--learning_rate', '-lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--optimizer', '-o', default='Adam', help='Optimizer')
parser.add_argument('--seed', type=int, default=100, help='Random seed')
parser.add_argument('--beta_1', type=float, default=0.9, help='Coefficient for EMA of Adam+')
parser.add_argument('--beta_2', type=float, default=0.999, help='Coefficient for EMA of Adam+')
parser.add_argument('--db', type=float, default=-3, help='dB threshold in Adam+')
parser.add_argument('--db_noise', type=float, default=-140, help='noise injection in Adam+')
parser.add_argument('--decay_rate', type=float, default=1.0, help='Decay rate')
parser.add_argument('--decay_steps', type=int, default=1, help='Decay steps')
parser.add_argument('--scheduler', type=int, default=1, help='LR scheduler type')
parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay L2 reg')
parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
args = parser.parse_args()

torch.manual_seed(args.seed)
generator = torch.Generator()
generator.manual_seed(args.seed)   # Same seed you use for model initialization

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, generator=generator, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, generator=generator)
test_loader = DataLoader(test_dataset, batch_size=64, generator=generator)


class GPS(torch.nn.Module):
    def __init__(self, channels: int, pe_dim: int, num_layers: int,
                 attn_type: str, attn_kwargs: Dict[str, Any]):
        super().__init__()

        self.node_emb = Embedding(28, channels - pe_dim)
        self.pe_lin = Linear(20, pe_dim)
        self.pe_norm = BatchNorm1d(20)
        self.edge_emb = Embedding(4, channels)

        self.convs = ModuleList()
        for _ in range(num_layers):
            nn = Sequential(
                Linear(channels, channels),
                ReLU(),
                Linear(channels, channels),
            )
            conv = GPSConv(channels, GINEConv(nn), heads=4,
                           attn_type=attn_type, attn_kwargs=attn_kwargs)
            self.convs.append(conv)

        self.mlp = Sequential(
            Linear(channels, channels // 2),
            ReLU(),
            Linear(channels // 2, channels // 4),
            ReLU(),
            Linear(channels // 4, 1),
        )
        self.redraw_projection = RedrawProjection(
            self.convs,
            redraw_interval=1000 if attn_type == 'performer' else None)

    def forward(self, x, pe, edge_index, edge_attr, batch):
        x_pe = self.pe_norm(pe)
        x = torch.cat((self.node_emb(x.squeeze(-1)), self.pe_lin(x_pe)), 1)
        edge_attr = self.edge_emb(edge_attr)

        for conv in self.convs:
            x = conv(x, edge_index, batch, edge_attr=edge_attr)
        x = global_add_pool(x, batch)
        return self.mlp(x)


class RedrawProjection:
    def __init__(self, model: torch.nn.Module,
                 redraw_interval: Optional[int] = None):
        self.model = model
        self.redraw_interval = redraw_interval
        self.num_last_redraw = 0

    def redraw_projections(self):
        if not self.model.training or self.redraw_interval is None:
            return
        if self.num_last_redraw >= self.redraw_interval:
            fast_attentions = [
                module for module in self.model.modules()
                if isinstance(module, PerformerAttention)
            ]
            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix()
            self.num_last_redraw = 0
            return
        self.num_last_redraw += 1


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
attn_kwargs = {'dropout': 0.5}
model = GPS(channels=64, pe_dim=8, num_layers=10, attn_type=args.attn_type,
            attn_kwargs=attn_kwargs).to(device)

optimizer = get_optimizer(model.parameters(), args.learning_rate, args)

# optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
if args.scheduler == 1:
    scheduler = torch.optim.lr_schedulerReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20, min_lr=0.000001
        )
elif args.scheduler == 2:
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.5)
elif args.scheduler == 3:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
else:
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs, gamma=1.0)


def train():
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        model.redraw_projection.redraw_projections()
        out = model(data.x, data.pe, data.edge_index, data.edge_attr,
                    data.batch)
        loss = (out.squeeze() - data.y).abs().mean()
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(loader):
    model.eval()

    total_error = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.pe, data.edge_index, data.edge_attr,
                    data.batch)
        total_error += (out.squeeze() - data.y).abs().sum().item()
    return total_error / len(loader.dataset)


results = pd.DataFrame([], columns=["epoch", "loss", "val_mae", "test_mae", "lr"])
csvname = "{}/output/{}_{}_{}_{}_{}_seed{}_{}_e{}_batch_{}_sch{}.csv".format(
    os.path.dirname(os.path.realpath(__file__)),
    "zinc",
    args.optimizer,
    "{:.1e}".format(args.learning_rate),
    args.beta_1,
    args.beta_2,
    args.seed,
    args.db,
    args.epochs,
    args.batch_size,
    args.scheduler,
)

for epoch in range(1, args.epochs+1):
    loss = train()
    val_mae = test(val_loader)
    test_mae = test(test_loader)
    if args.scheduler == 1:
        scheduler.step(loss) #val_mae
        # scheduler.step(val_mae) #val_mae
    elif args.scheduler == 2 or args.scheduler == 3:
        scheduler.step()
    else:
        pass

    current_lr = scheduler.get_last_lr()[0]
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val: {val_mae:.4f}, '
          f'Test: {test_mae:.4f}, lr: {current_lr:.6f}')

    log_item = {
        "epoch": epoch,
        "loss": loss,
        "val_mae": val_mae,
        "test_mae": test_mae,
        "lr": current_lr,
        "opt": args.optimizer
    }
    df_item = pd.DataFrame([log_item])
    results = pd.concat([results, df_item], ignore_index=True)
    # results = results.reset_index(drop=True)
    results.to_csv(csvname)
