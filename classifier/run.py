import os
import sys
import time
import networkx as nx
import scipy.io as sio
import scipy.sparse as sp
from scipy.spatial import distance_matrix
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation

import pickle

import re
numeric_const_pattern = r"""
     [-+]? 
     (?:
         (?: \d* \. \d+ )
         |
         (?: \d+ \.? )
     )
     (?: [Ee] [+-]? \d+ ) ?
     """
rx = re.compile(numeric_const_pattern, re.VERBOSE)

from IPython.display import clear_output
from IPython.display import Video

from scipy.stats.stats import pearsonr
import subprocess
from datetime import datetime

bash_dir= "bash"
output_dir = "output"
config_file = "zinc_gps_config.csv"

env = "mnist"
epochs = 200
seed = "51"
db=-30
decay_rate = 1
decay_steps=30
batch_size=100
weight_decay=0.001
scheduler = 0 # constant learning rate no scheduled decay
max_threads = 30
df_all = pd.read_csv(os.path.join(os.path.dirname(__file__), bash_dir, config_file),index_col=0) 
df_all = df_all.sort_index()
df_all
df = df_all
#  df=df_all.iloc[[6,7,0,20,17,10,9,1,13,18]]
# df["lr"]=0.001
# df.loc[df["opt"]=="Lion", "lr"]=0.0001
# df.loc[df["opt"]=="Adam41", "lr"]=0.0001
# df_config.loc[df_config["opt"]=="Adam4", "lr"]=0.001
# df
df = df[df['opt'] == "Adam2"]
running_procs = []

for i, row in df.iterrows():
    opt = row["opt"]
    beta1 = row["beta1"]
    beta2 = row["beta2"]
    lr = row["lr"]
    # lr = 0.0001

    outname = f"{env}_epochs{epochs}_seed{seed}_{beta1}_{beta2}_{opt}_{lr}.txt"
    log_path = f"{output_dir}/{outname}"

    cmd = [
        "python3.10", "main.py",
        f"--seed={seed}",
        f"--learning_rate={lr}",
        f"--epochs={epochs}",
        f"--optimizer={opt}",
        f"--beta_1={beta1}",
        f"--beta_2={beta2}",
        f"--db={db}",
        f"--weight_decay={weight_decay}",
        f"--batch_size={batch_size}",
        f"--decay_steps={decay_steps}",
        f"--decay_rate={decay_rate}",
        f"--model={env}"
    ]

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}][Submitting]: {' '.join(cmd)} > {log_path}")

    with open(log_path, "w") as outfile:
        cmd = ["nohup"] + cmd
        proc = subprocess.Popen(cmd, stdout=outfile, stderr=outfile, preexec_fn=os.setpgrp)
        running_procs.append(proc)

    # Limit number of concurrent jobs
    while len(running_procs) >= max_threads:
        for p in running_procs:
            if p.poll() is not None:  # finished
                running_procs.remove(p)
        time.sleep(30)  # wait a bit before checking again

# Wait for remaining jobs to finish
for p in running_procs:
    p.wait()
