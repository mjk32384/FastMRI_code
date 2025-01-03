import numpy as np
import os, sys

if os.getcwd() + '/utils/model/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/model/')
from utils.learning.train_part import train

report_interval = 100

epoch = 10
batch_size = 1
lr = 1e-3

net_name = 'test_6125_59'
cascade = 6
chans = 12
sens_chans = 5
acc_weight = "{5:1/2,9:1/2}"


print(f"Running training with lr={lr}, epochs={epoch}, cascade={cascade}, chans={chans}, sens_chans={sens_chans}, acc_weight={acc_weight}")


os.system(f'python -W ignore train.py -b {batch_size} -e {epoch} -l {lr} -r {report_interval} -n {net_name}\
    --cascade {cascade} --chans {chans} --sens_chans {sens_chans} --acc_weight {acc_weight}')