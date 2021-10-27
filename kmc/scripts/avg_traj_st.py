#!/usr/bin/python

import sys 
from numpy import *

####################
#Average over kmc trajectories
####################

ntraj = 100

kT = 0.2
h = -10.0
kswap=1000000.0

Ns = int(sys.argv[1])

N = 3*Ns*Ns-3*Ns+1

mypopdir= '/home/frechette/ionex/kmc/data/J=0.000000/Ns=%d/kT=%f/h=%f/k_swap=%f/pops/' % (Ns, kT, h, kswap)

data = [loadtxt(mypopdir + 'pops_st_seed=%d.txt' % (i+1)) for i in range(ntraj)]

#short-time data
xmin = min(line[1:,0].min() for line in data)
xmax = max(line[1:,0].max() for line in data)

x_points = linspace(log10(xmin), log10(xmax), 200)

interpolated = [interp(x_points, log10(d[1:,0]), d[1:,1]) for d in data]

averages = [average(x) for x in zip(*interpolated)]
stdevs = [std(x)/ntraj for x in zip(*interpolated)]

savetxt(mypopdir + 'pops_avg_st.txt', c_[x_points, averages, stdevs])

