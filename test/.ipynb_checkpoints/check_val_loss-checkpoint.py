import numpy as np

net_name = 'test_894_varnet_2'

val_loss_log = np.load('../../result/'+ net_name +'/val_loss_log' + '.npy')
for epoch in val_loss_log:
    print("Epoch: " + "%02d"%int(epoch[0]) + ", Loss: " + "%0.4f"%epoch[1])