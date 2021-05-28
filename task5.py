import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

def simulation(distro, dt, times, totalx, nx, mfp, vel, tsteps, eps):
    r = [0.0, 0.0, 0.0]

    dx = totalx / nx

    ## velocity before collision
    a0 = 1
    b0 = 0
    c0 = 0

    ## velocity after collision
    a = 0
    b = 0
    c = 0

    curtime = 0

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

        curtime += l / vel

        if abs(r[0]) < totalx and abs(r[1]) < totalx:
            for i in range(len(times)):
                if abs(curtime - times[i]) < dt:
                    indx = int(r[0] / dx) + nx
                    indy = int(r[1] / dx) + nx
                    distro[i][indx][indy] += 1

        a0, b0, c0 = a, b, c
    return distro

def analytical(x, t, dx, dy):
    D = 1 / 3
    u = (4 * np.pi * D * t)
    return 1 / u  * np.exp(-x**2 / u) * dx * dy

def plot(distro, totalx, nx, times_ind, t, title):
    dx = totalx / nx

    sns.set(context='notebook', style='darkgrid')
    fig, ax = plt.subplots( figsize=(8, 8))
    fig.suptitle(title)

    x = np.arange(-nx, nx, 1) * dx

    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'Normalized Number of Particles $K$')

    ax.plot(x, distro, label="Numerical Solution")
    ax.plot(x, analytical(x, t, dx, dx),
            label="Analytical Solution")
    ax.legend()

    # fig.savefig("media/task5.pdf")
    fig.savefig("media/task5_{0}.svg".format(times_ind))
    fig.savefig("media/task5_{0}.png".format(times_ind), dpi=300)
    # plt.show()

tsteps = 200
npart = 200000
eps = 0.0001
mfp = 1
vel = 1
deltaT = 10 * mfp / vel

totalx = 30
nx = 30
dt = 0.6

times = np.arange(40, 121, 40)
# times = [40]

distro = np.zeros((len(times), nx * 2, nx * 2))

for i in range(npart):
    simulation(distro, dt, times, totalx, nx, mfp, vel,
            tsteps, eps)

for i in range(len(times)):
    print("Total particles at time t = ", times[i],
            " -> N = ", np.sum(distro[i]))
    distro[i] /= np.sum(distro[i])

for i in range(len(times)):
    plot(distro[i][nx - 1], totalx, nx, i + 10, times[i],
            "Space Distribution Function: Cross Section x = 0, Time = {0}".format(times[i]))

for i in range(len(times)):
    plot(distro[i][:][nx - 1], totalx, nx, i + 20, times[i],
            "Space Distribution Function: Cross Section y = 0, Time = {0}".format(times[i]))
