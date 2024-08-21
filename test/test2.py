import numpy as np
from pathlib import Path
import h5py
import matplotlib.pyplot as plt

path = Path('../../../home/Data/train')
image_file = sorted(list(Path(path / "kspace").iterdir()))[0]
with h5py.File(image_file, "r") as hf:
    arr = np.array(hf.get('kspace'))

print(np.min(arr))
x = np.log(np.abs(arr.flatten()) + 10**(-8))
#y = np.angle(arr.flatten())
#plt.xscale('log')
plt.hist(x, bins = 100)
plt.savefig('fig.png')
#print(np.max(x))