import sys
from math import *
import numpy as np
import scipy.spatial as spatial

bc = 'nc'
kT = 0.2
h = 0.0
c = 0.5
shell_layers = 1

Ns = int(sys.argv[1])

if Ns==8:
  Kvals = [1.0,2.0,3.0,4.0,4.2,4.4,4.6,4.8,5.0,5.2,5.4,5.6,5.8,6.0,6.2,6.4]
elif Ns==10:
  Kvals = [1.0,2.0,3.0,5.0,5.2,5.4,5.6,5.8,6.0,6.2,6.4,6.6,6.8]
elif Ns==12:
  Kvals = [5.4,5.6,5.8,6.0,6.2,6.4,6.6,7.0,7.2,7.4,7.6,7.8,7.9,8.0,8.2,8.4,8.6,8.8,9.0,9.2]
elif Ns==16:
  Kvals = [5.0,6.0,7.0,8.0,9.0,9.2,9.4,9.6,9.7,9.8,9.9,10.0,10.1,10.2,10.3,10.4,10.5,10.6,10.7,10.8,10.9,11.0]
elif Ns==24:
  Kvals = [11.0,12.0,13.0,14.0,15.0,15.2,15.4,15.6,15.8,16.0,16.2,16.4,16.6,16.7,16.8,16.81,16.82,16.83,16.84,16.85,16.86,16.87,16.88,16.89,16.9,17.0,18.0,19.0,20.0]
else:
  print('Size not supported.')
nstiffvals = len(Kvals)

avg_u2 = np.zeros(nstiffvals)
avg_s2 = np.zeros(nstiffvals)
avg_unc = np.zeros(nstiffvals)
avg_s1 = np.zeros(nstiffvals)
avg_u1 = np.zeros(nstiffvals)

sus_u2 = np.zeros(nstiffvals)
sus_s2 = np.zeros(nstiffvals)
sus_unc = np.zeros(nstiffvals)
sus_s1 = np.zeros(nstiffvals)
sus_u1 = np.zeros(nstiffvals)
sus_mod = np.zeros(nstiffvals)
sus_uns = np.zeros(nstiffvals)

cnter=0
for K in Kvals:
  data = np.loadtxt('/home/frechette/ionex/kmc/equil/cg_spin_pops_Kshell=%f.txt' % K)
  avg_u2[cnter] = np.average(data[:,1])
  avg_s2[cnter] = np.average(data[:,2])
  avg_unc[cnter] = np.average(data[:,3])
  avg_s1[cnter] = np.average(data[:,4])
  avg_u1[cnter] = np.average(data[:,5])

  sus_u2[cnter] = np.average(data[:,1]**2)-avg_u2[cnter]**2
  sus_s2[cnter] = np.average(data[:,2]**2)-avg_s2[cnter]**2
  sus_unc[cnter] = np.average(data[:,3]**2)-avg_unc[cnter]**2
  sus_s1[cnter] = np.average(data[:,4]**2)-avg_s1[cnter]**2
  sus_u1[cnter] = np.average(data[:,5]**2)-avg_u1[cnter]**2

  sus_mod[cnter] = np.average((data[:,2]+data[:,4])**2)-np.average(data[:,2]+data[:,4])**2
  sus_uns[cnter] = np.average((data[:,1]+data[:,5])**2)-np.average(data[:,1]+data[:,5])**2
  cnter = cnter + 1

np.savetxt('sus_Ns=%d.txt' % Ns, np.c_[Kvals, avg_u2, avg_s2, avg_unc, avg_s1, avg_u1, sus_u2, sus_s2, sus_unc, sus_s1, sus_u1, sus_mod, sus_uns])


