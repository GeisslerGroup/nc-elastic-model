#!/usr/bin/python

import sys 
from numpy import *

####################
#Average over kmc trajectories
####################

ntraj = 100

kT = 1.5
h = -100.0
kswap=1.0

Ns = int(sys.argv[1])

N = Ns**3

mypopdir= '/home/frechette/ionex/kmc/3d/data/J=0.000000/nx=%d_ny=%d_nz=%d/kT=%f/h=%f/k_swap=%f/pops/' % (Ns, Ns, Ns, kT, h, kswap)

data = [loadtxt(mypopdir + 'pops_seed=%d.txt' % (i+1)) for i in range(ntraj)]

#Long-time data
xmin = min(line[:,0].min() for line in data)
xmax = max(line[:,0].max() for line in data)

x_points = linspace(xmin, xmax, 300)

interpolated = [interp(x_points, d[:,0], d[:,1]) for d in data]

averages = [average(x) for x in zip(*interpolated)]
stdevs = [std(x)/ntraj for x in zip(*interpolated)]

savetxt(mypopdir + 'pops_avg.txt', c_[x_points, averages, stdevs])