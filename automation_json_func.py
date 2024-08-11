import numpy as np
import os, sys
import json
from utils.learning.train_part import train


def auto(args):
    if os.getcwd() + '/utils/model/' not in sys.path:
        sys.path.insert(1, os.getcwd() + '/utils/model/')
    
    params = {'report_interval':args['report_interval'],
              'epoch':args['epoch'],
              'batch_size':args['batch_size'],
              'learning_rate':args['lr'],
              'net_name':args['net_name'],
              'cascade':args['cascade'],
              'chans':args['chans'],
              'sens_chans':args['sens_chans'],
              'acc_weight':args['acc_weight'],
              'previous_model':args['previous_model']}
    
    
    #print(json.dumps(params, indent = 4))
    with open('params.json', 'w') as f:
        json.dump(params, f, indent = 4)
    
    os.system(f'python -W ignore train_json.py')
    os.system(f'python -W ignore reconstruct_json.py')
    os.system(f'python -W ignore leaderboard_eval_json.py')