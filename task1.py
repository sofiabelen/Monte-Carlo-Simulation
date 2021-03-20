import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.optimize import curve_fit
import random

## mfp: mean free path
def lk(mfp):
    rk = random.random()
    return -mfp * np.log(1 - rk)

def thetak():
    rk = random.random()
    return np.arccos(1 - 2 * rk)

def phik():
    rk = random.random()
    return 2 * np.pi * rk

def simulation(tsteps=100, eps=0.0001, seed=10):
    ## Set a seed: for debugging purposes
    random.seed(seed)
    mfp = 1
    r = [0.0, 0.0, 0.0]
    vwork = [1.0, 0.0, 0.0]
    vnext = [0.0, 0.0, 0.0]
    msd = np.zeros(tsteps)

    for t in range(tsteps):
        l = lk(mfp)
        theta = thetak()
        phi = phik()

        ## velocity in relative frame of reference
        u = [np.sin(theta) * np.cos(phi),\
                np.sin(theta) * np.sin(phi),\
                np.cos(theta)]

        ## velocity in absolute frame of reference
        if(abs(u[2] - 1) < eps):
            vnext = u
        else:
            mu = np.cos(theta)
            vnext[0] = mu * u[0] - (u[1] * np.sin(phi) -\
                    u[0] * u[2] * np.cos(phi)) *\
                    np.sqrt((1 - mu**2) / (1 - u[2]**2))
            vnext[1] = mu * u[1] + (u[0] * np.sin(phi) +\
                    u[1] * u[2] * np.cos(phi)) *\
                    np.sqrt((1 - mu**2) / (1 - u[2]**2))
            vnext[2] = mu * u[2] - (1 - u[2]**2) *\
                    np.cos(phi)\
                    * np.sqrt((1 - mu**2) / (1 - u[2]**2))

        for j in range(3):
            r[j] += vnext[j] * l
            msd[t] += r[j]**2

        vwork = vnext
    return msd

def plot(msd):
    tsteps = len(msd)
    linestart = int(0.7 * tsteps)

    def line(x, a, b):
        return a*x + b

    popt, pcov = curve_fit(f=line,\
            xdata=np.arange(linestart, tsteps),\
            ydata=msd[linestart:])

    sns.set(context='notebook', style='darkgrid')
    fig, ax = plt.subplots(figsize=(8, 8))
    
    time = np.arange(linestart, tsteps)
    ax.plot(np.arange(tsteps), msd)
    ax.plot(time, line(time, *popt),\
            label=r'$D = %.3f$'%(popt[0] / 6))
    
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$<r^2(t)>$')
    
    ax.legend()
    
    fig.savefig("media/task1.pdf")
    fig.savefig("media/task1.svg")
    fig.savefig("media/task1.png", dpi=200)
    plt.show()

tsteps = 100000
npart = 10
eps = 0.0001
seed = 10
msd = np.zeros((npart, tsteps))

for i in range(npart):
    msd[i] = simulation(tsteps, eps, seed)

plot(np.average(msd, axis=0))
