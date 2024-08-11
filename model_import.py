import torch
checkpoint = torch.load('../result/test_894_varnet/checkpoints/'+'model.pt', map_location='cpu')
print(checkpoint['epoch'], checkpoint['best_val_loss'].item())