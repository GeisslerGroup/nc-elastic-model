#!/usr/bin/python

from numpy import *

####################
#Average over kmc trajectories
####################

#Ising at different temperatures

ntraj = 10000

temp = 6.0
h = 0.5
nx=12
ny=14

for i in range(ntraj):

    data = loadtxt('pops/bulk/elastic/kT=%f/h=%f/nx=%d_ny=%d/mean_field/kmc_pop_seed=%d_dyn=1.txt' % (temp, h, nx, ny, (i+1)))

    xmin = min(line[:,0].min() for line in data)
    xmax = max(line[:,0].max() for line in data)

    x_points = linspace(xmin, xmax, 10000)

    interpolated = [interp(x_points, d[:,0], d[:,1]) for d in data]

    averages = [average(x) for x in zip(*interpolated)]
    stdevs = [std(x)/ntraj for x in zip(*interpolated)]


    #savetxt('pops/bulk/elastic/kT=%f/h=%f/nx=%d_ny=%d/mc_metro/kmc_pop_avg.txt' % (temp, h, nx, ny), c_[x_points, averages, stdevs])
    savetxt('pops/bulk/elastic/kT=%f/h=%f/nx=%d_ny=%d/mean_field/kmc_pop_avg_dyn=1.txt' % (temp, h, nx, ny), c_[x_points, averages, stdevs])

