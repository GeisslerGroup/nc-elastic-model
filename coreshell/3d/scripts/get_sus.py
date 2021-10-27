import sys
from math import *
import numpy as np
import scipy.spatial as spatial

bc = 'nc'
kT = 1.5
h = 0.0
c = 0.5
shell_layers = 1

Ns = int(sys.argv[1])

if (Ns==8 or Ns==10 or Ns==12):
  Kvals = [0.5,0.75,1.0,1.1,1.2,1.25,1.3,1.4,1.5,1.6,1.7,1.75,1.8,1.9,2.0,2.1,2.2,2.25,2.3,2.4,2.5,2.6,2.7,2.75,2.8,2.9,3.0,4.0]
elif (Ns==14 or Ns==16):
  Kvals = [2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.0,3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8,3.9,4.0]

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
  data = np.loadtxt('/home/frechette/ionex/kmc/equil/3d/confs/%s/3d/stiffshell/kT=%f/h=%f/Ns=%d/c=%f/shell_layers=%d/Kshell=%f/cg_spin_pops.txt' % (bc, kT, h, Ns, c, shell_layers, K))
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

np.savetxt('sus3d_Ns=%d.txt' % Ns, np.c_[Kvals, avg_u2, avg_s2, avg_unc, avg_s1, avg_u1, sus_u2, sus_s2, sus_unc, sus_s1, sus_u1, sus_mod, sus_uns])


