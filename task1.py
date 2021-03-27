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

def simulation(mfp=1, vel=1, tsteps=1000, eps=0.0001):
    r = [0.0, 0.0, 0.0]

    ## velocity before collision
    a0 = 1
    b0 = 0
    c0 = 0

    ## velocity after collision
    a = 0
    b = 0
    c = 0

    msd = np.zeros(tsteps)
    time = np.zeros(tsteps)

    for t in range(tsteps):
        l = lk(mfp)
        theta = thetak()
        phi = phik()

        if(abs(c - 1) < eps):
            a = np.sin(theta) * np.cos(phi)
            b = np.sin(theta) * np.sin(phi)
            c = np.cos(theta)
        else:
            mu = np.cos(theta)
            a = mu * a0 - (b0 * np.sin(phi)\
                    - a0 * c0 * np.cos(phi))\
                    * np.sqrt((1 - mu**2) / (1 - c0**2))
            b = mu * b0 + (a0 * np.sin(phi)\
                    + b0 * c0 * np.cos(phi))\
                    * np.sqrt((1 - mu**2) / (1 - c0**2))
            c = mu * c0 - (1 - c0**2) * np.cos(phi)\
                    * np.sqrt((1 - mu**2) / (1 - c0**2))

        v = [a, b, c]
        for j in range(3):
            r[j] += v[j] * l
            msd[t] += r[j]**2

        time[t] = l / vel
        if t > 0:
            time[t] += time[t - 1]

        a0, b0, c0 = a, b, c
    return msd, time

def plot(msd, totaltime, vel, mfp):
    tsteps = len(msd)
    deltaT = totaltime / tsteps

    sns.set(context='notebook', style='darkgrid')
    fig, ax = plt.subplots(figsize=(8, 8))
    
    time = np.arange(0, totaltime, deltaT)
    ax.plot(time, msd)
    D = mfp * vel / 3
    ax.plot(time, time * 6 * D,\
            label=r'$\Delta r^2(t) = 6Dt,\quad D = \frac{\lambda v_T}{3} = %.3f$'%(D))
    
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$<r^2(t)>$')
    
    ax.legend()
    
    fig.savefig("media/task1.pdf")
    fig.savefig("media/task1.svg")
    fig.savefig("media/task1.png", dpi=200)
    plt.show()

def average(msd, time, deltaT):
    npart = len(msd)
    tsteps = len(msd[0])
    totaltime = 1000000000

    for i in range(npart):
        totaltime = min(totaltime, time[i][tsteps - 1])

    tsteps2 = int(totaltime / deltaT)
    msdAvg = np.zeros(tsteps2)

    for i in range(npart):
        t = 0
        k = 0
        for j in range(tsteps):
            if time[i][j] > t and k < tsteps2:
                msdAvg[k] += msd[i][j]
                t += deltaT
                k += 1

    msdAvg /= npart
    return msdAvg, totaltime

tsteps = 1000
npart = 1000
eps = 0.0001
mfp = 1
vel = 1
deltaT = 10 * mfp / vel
msd = np.zeros((npart, tsteps))
time = np.zeros((npart, tsteps))

for i in range(npart):
    msd[i], time[i] = simulation(mfp, vel, tsteps, eps)

msdAvg, totaltime = average(msd, time, deltaT)

plot(msdAvg, totaltime, vel, mfp)
