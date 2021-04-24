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

def plot(distro, totalx, nx, times_ind, times_i):
    dx = totalx / nx

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    X = np.arange(-nx, nx, 1) * dx
    Y = np.arange(-nx, nx, 1) * dx
    X, Y = np.meshgrid(X, Y)

    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_zlabel(r'Number of particles $K$')
    fig.suptitle(r'Time t = $%.1f$'%times_i)

    ax.plot_surface(X, Y, distro, linewidth=0,
            antialiased=False)

    # fig.savefig("media/task4_{0}.pdf".format(times_ind))
    # fig.savefig("media/task4_{0}.svg".format(times_ind))
    # fig.savefig("media/task4_{0}.png".format(times_ind), dpi=300)
    fig.savefig("media/gif/task4_{0}.png".format(times_ind))
    # plt.show()

tsteps = 100
npart = 10000
eps = 0.0001
mfp = 1
vel = 1
deltaT = 10 * mfp / vel

totalx = 30
nx = 30
dt = 0.6

times = np.arange(0, 101, 10)
# times = [30]

distro = np.zeros((len(times), nx * 2, nx * 2))

for i in range(npart):
    simulation(distro, dt, times, totalx, nx, mfp, vel, tsteps, eps)

for i in range(len(times)):
    print("Total particles at time t = ", times[i],
            " -> N = ", np.sum(distro[i]))

# for i in range(nx * 2):
#     for j in range(nx * 2):
#         if distro[0][i][j] != 0:
#             # print(distro[0][i][j], end = ' ')
#             print(i, j)
#     # print("")

for i in range(len(times)):
    plot(distro[i], totalx, nx, i, times[i])
