import os

report_interval = 100

epoch = 10
batch_size = 1
lr = 1e-3

net_name = 'test_Varnet_random_mask'
cascade = 6
chans = 12
sens_chans = 5


print(f"Running training with lr={lr}, epochs={epoch}, cascade={cascade}, chans={chans}, sens_chans={sens_chans}")

os.system(f"python train_model.py -b {batch_size} -e {epoch} -l {lr} -r {report_interval} -n {net_name} --cascade {cascade} --chans {chans} --sens_chans {sens_chans}")