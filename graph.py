import matplotlib.pyplot as plt
import numpy as np

def draw_val_loss(data):
    fig = plt.figure(figsize=(10, 13))
    grid = plt.GridSpec(4, 3, hspace= 0.4, wspace= 0.4)
    ax = []
    for i in range(9):
        if i == 0:
            ax.append(fig.add_subplot(grid[i//3, i%3]))
        else:
            ax.append(fig.add_subplot(grid[i//3, i%3], sharex = ax[0], sharey = ax[0]))
        ax[i].plot(range(1, len(data)+1), data[:, i+1])
        ax[i].set_title(f"acc{i+2}")
    total_ax = fig.add_subplot(grid[3:,:])
    total_ax.plot(range(2, 11), data[-1,1:])
    total_ax.set_xlabel("acc")