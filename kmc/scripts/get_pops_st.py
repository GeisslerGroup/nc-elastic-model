import sys
from math import *
import numpy as np

Ns = 8
kT = 0.2
h = -10
kswap=1000000.0

Ns = int(sys.argv[1])
seed=int(sys.argv[2])

N=3*Ns*Ns-3*Ns+1
nframe = N

my_dict = {'Ar':1, 'S':-1}

mydir= '/home/frechette/ionex/kmc/data/J=0.000000/Ns=%d/kT=%f/h=%f/k_swap=%f/confs/seed=%03d/' % (Ns, kT, h, kswap, seed)
mypopdir= '/home/frechette/ionex/kmc/data/J=0.000000/Ns=%d/kT=%f/h=%f/k_swap=%f/pops/' % (Ns, kT, h, kswap)

spin_data = np.zeros(nframe)
times = np.zeros(nframe)

######################
print("Getting pops for seed %d." % seed)
for i in range(nframe):  

  myfile = mydir + 'shortt_%04d.xyz' % i 
  conf = open(myfile)

  line = conf.readline()
  n = int(line.strip())
  line2 = conf.readline()
  t = float(line2.strip()[5:])
  times[i] = t
  coords = np.loadtxt(myfile, skiprows=2, usecols=(1,2))
  atom_types = np.loadtxt(myfile, dtype=str, skiprows=2, usecols=(0)).tolist()
  spins = np.array([my_dict[atom] for atom in atom_types]) 

  spin_data[i] = 1.0*np.count_nonzero(spins==1)/(1.0*n)

np.savetxt(mypopdir + 'pops_st_seed=%d.txt' % seed, np.c_[times,spin_data])
