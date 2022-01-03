import numpy as np
import matplotlib.pyplot as plt

def infinity():
    """
    :return: a generator from 0 to inf
    """
    i = 0
    while True:
        i += 1
        yield i

def simulated_graphs(pl, name):
    pl.sort(key=lambda x: x[1])
    for li in pl:
        if li[1] < 0:
            li[1] = -5
        else:
            li[1] = round(li[1], 2)
    pl = np.array(pl)
    x1, y1 = pl.T
    i = np.array(['inf'])
    y = np.setdiff1d(y1, i)
    y = y.astype(np.float)
    x1 = x1.astype(np.float)
    plt.scatter(x1, y1)
    plt.title('SSP Simulated-Annealing:' + name)
    plt.xlabel('iteration')
    plt.ylabel('utility')
    xtick = np.arange(x1.min(), x1.max(), 250)
    plt.xticks(xtick)
    plt.show()