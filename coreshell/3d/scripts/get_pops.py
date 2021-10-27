import sys
from math import *
import numpy as np

nframe = 10000

bc = 'nc'
kT = 1.5
h = 0.0
c = 0.5
shell_layers = 1

Ns = int(sys.argv[1])
Kshell_ind = int(sys.argv[2])-1

if (Ns==8 or Ns==10 or Ns==12):
  Kvals = [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,0.5,0.75,1.25,1.5,1.75,2.25,2.5,2.75,1.1,1.2,1.3,1.4,1.6,1.7,1.8,1.9,2.1,2.2,2.3,2.4,2.6,2.7,2.8,2.9]
elif (Ns==14 or Ns==16):
  Kvals = [2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.0,3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8,3.9,4.0]

Kshell = Kvals[Kshell_ind]

my_dict = {'H':-2, 'He':-1, 'Li':0, 'Be':1, 'B':2}

mydir= '/home/frechette/ionex/kmc/equil/3d/confs/%s/3d/stiffshell/kT=%f/h=%f/Ns=%d/c=%f/shell_layers=%d/Kshell=%f/' % (bc, kT, h, Ns, c, shell_layers, Kshell)

spin_data = np.zeros((nframe, 5))

######################
print('Ns: %d, shell_layers: %d, Kshell: %f' % (Ns, shell_layers, Kshell))
for i in range(nframe):  

  myfile = mydir + 'cg_%05d.xyz' % i 
  conf = open(myfile)

  line = conf.readline()
  n = int(line.strip())
  atom_types = np.loadtxt(myfile, dtype=str, skiprows=2, usecols=(0)).tolist()
  spins = np.array([my_dict[atom] for atom in atom_types]) 

  spin_data[i,0] = 1.0*np.count_nonzero(spins==-2)/(1.0*n)
  spin_data[i,1] = 1.0*np.count_nonzero(spins==-1)/(1.0*n)
  spin_data[i,2] = 1.0*np.count_nonzero(spins==0)/(1.0*n)
  spin_data[i,3] = 1.0*np.count_nonzero(spins==1)/(1.0*n)
  spin_data[i,4] = 1.0*np.count_nonzero(spins==2)/(1.0*n)

np.savetxt(mydir + 'cg_spin_pops.txt', np.c_[np.arange(0,nframe),spin_data])
