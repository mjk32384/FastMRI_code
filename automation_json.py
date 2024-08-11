import numpy as np
import os, sys
import json

if os.getcwd() + '/utils/model/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/model/')
from utils.learning.train_part import train

report_interval = 100

epoch = 1
batch_size = 1
lr = 1e-3

net_name = 'temp'
cascade = 3
chans = 5
sens_chans = 3
acc_weight = {5:1/2,9:1/2}
previous_model = False

params = {'report_interval':report_interval,
          'epoch':epoch,
          'batch_size':batch_size,
          'learning_rate':lr,
          'net_name':net_name,
          'cascade':cascade,
          'chans':chans,
          'sens_chans':sens_chans,
          'acc_weight':acc_weight,
          'previous_model':previous_model}


#print(json.dumps(params, indent = 4))
with open('params.json', 'w') as f:
    json.dump(params, f, indent = 4)

os.system(f'python -W ignore train_json.py')
os.system(f'python -W ignore reconstruct_json.py')
os.system(f'python -W ignore leaderboard_eval_json.py')