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

data = [loadtxt(mypopdir + 'cg_spin_pops_seed=%d.txt' % (i+1)) for i in range(ntraj)]

#Long-time data
xmin = min(line[:,0].min() for line in data)
xmax = max(line[:,0].max() for line in data)

x_points = linspace(xmin, xmax, 10000)

int1 = [interp(x_points, d[:,0], d[:,1]) for d in data]
int2 = [interp(x_points, d[:,0], d[:,2]) for d in data]
int3 = [interp(x_points, d[:,0], d[:,3]) for d in data]
int4 = [interp(x_points, d[:,0], d[:,4]) for d in data]
int5 = [interp(x_points, d[:,0], d[:,5]) for d in data]

avg1 = [average(x) for x in zip(*int1)]
avg2 = [average(x) for x in zip(*int2)]
avg3 = [average(x) for x in zip(*int3)]
avg4 = [average(x) for x in zip(*int4)]
avg5 = [average(x) for x in zip(*int5)]

savetxt(mypopdir + 'cg_avg.txt', c_[x_points, avg1, avg2, avg3, avg4, avg5])

