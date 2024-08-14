import numpy as np
import os, sys
import json
if os.getcwd() + '/utils/model/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/model/')
    
from utils.learning.train_part import train
import subprocess


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
              'previous_model':args['previous_model'],
              # added from here
              'mask_mode':args['mask_mode'],
              'use_SSIM_mask_train':args['use_SSIM_mask_train']}
                # to here
    
    
    #print(json.dumps(params, indent = 4))
    with open('params.json', 'w') as f:
        json.dump(params, f, indent = 4)
    try:
        subprocess.check_call('python -W ignore train_json.py', shell = True, stderr=sys.stderr, stdout=sys.stdout, stdin = sys.stdin)
    except subprocess.CalledProcessError:
        print ("Error Occured")
        raise Exception("Cmd Command Error - Training")
    try:
        subprocess.check_call('python -W ignore reconstruct_json.py', shell = True, stderr=sys.stderr, stdout=sys.stdout, stdin = sys.stdin)
    except subprocess.CalledProcessError:
        print ("Error Occured")
        raise Exception("Cmd Command Error - Reconstructing")
    result = ""
    try:
        result = subprocess.check_output('python -W ignore leaderboard_eval_json.py', shell = True)
        print(result.decode("utf-8"))
    except subprocess.CalledProcessError:
        print ("Error Occured")
        raise Exception("Cmd Command Error - Leaderboard Eval")
    return result.decode("utf-8")