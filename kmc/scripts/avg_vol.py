#!/usr/bin/python

import sys 
from numpy import *

####################
#Average over kmc trajectories
####################

ntraj = 100

kT = 0.2
h = -10.0
kswap=1.0

Ns = int(sys.argv[1])

N = 3*Ns*Ns-3*Ns+1

mypopdir= '/home/frechette/ionex/kmc/data/J=0.000000/Ns=%d/kT=%f/h=%f/k_swap=%f/pops/' % (Ns, kT, h, kswap)

data = [loadtxt(mypopdir + 'rmsvol_seed=%d.txt' % (i+1)) for i in range(ntraj)]

#Long-time data
xmin = min(line[:,0].min() for line in data)
xmax = max(line[:,0].max() for line in data)

x_points = linspace(xmin, xmax, 10000)

interpolated = [interp(x_points, d[:,0], d[:,1]) for d in data]

averages = [average(x) for x in zip(*interpolated)]
stdevs = [std(x)/ntraj for x in zip(*interpolated)]

savetxt(mypopdir + 'rmsvol_avg.txt', c_[x_points, averages, stdevs])
