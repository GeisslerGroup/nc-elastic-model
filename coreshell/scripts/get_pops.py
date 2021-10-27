import sys
from math import *
import numpy as np

nframe = 100000

bc = 'nc'
kT = 0.2
h = 0.0
c = 0.5
shell_layers = 1

Ns = int(sys.argv[1])
Kshell_ind = int(sys.argv[2])-1

if Ns==8:
  Kvals = [1.0,2.0,3.0,4.0,4.2,4.4,4.6,4.8,5.0,5.2,5.4,5.6,5.8,6.0,6.2,6.4]
elif Ns==10:
  Kvals = [1.0,2.0,3.0,5.0,5.2,5.4,5.6,5.8,6.0,6.2,6.4,6.6,6.8,7.0,7.2,8.0]
elif Ns==12:
  Kvals = [4.0,5.0,6.2,5.4,5.6,5.8,6.0,6.2,6.4,6.6,7.0,7.2,7.4,7.6,7.8,7.9,8.0,8.2,8.4,8.6,8.8,9.0,9.2]
elif Ns==16:
  Kvals = [5.0,6.0,7.0,8.0,9.0,9.2,9.4,9.5,9.6,9.7,9.8,9.9,10.0,10.1,10.2,10.3,10.4,10.5,10.6,10.7,10.8,10.9,11.0]
elif Ns==24:
  Kvals = [11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0,19.0,20.0,15.2,15.4,15.6,15.8,16.2,16.4,16.6,16.8,16.7,16.9,16.81,16.82,16.83,16.84,16.85,16.86,16.87,16.88,16.89]
else:
  print('Size not supported.')

Kshell = Kvals[Kshell_ind]

my_dict = {'H':-2, 'He':-1, 'Li':0, 'Be':1, 'B':2}

mydir= '/home/frechette/ionex/kmc/equil/confs/%s/stiffshell/kT=%f/h=%f/Ns=%d/c=%f/shell_layers=%d/Kshell=%f/' % (bc, kT, h, Ns, c, shell_layers, Kshell)

spin_data = np.zeros((nframe, 5))

######################
print('Ns: %d, shell_layers: %d, Kshell: %f' % (Ns, shell_layers, Kshell))
for i in range(nframe):  

  myfile = mydir + 'cg_%05d.xyz' % i 
  conf = open(myfile)

  line = conf.readline()
  n = int(line.strip())
  coords = np.loadtxt(myfile, skiprows=2, usecols=(1,2))
  atom_types = np.loadtxt(myfile, dtype=str, skiprows=2, usecols=(0)).tolist()
  spins = np.array([my_dict[atom] for atom in atom_types]) 

  spin_data[i,0] = 1.0*np.count_nonzero(spins==-2)/(1.0*n)
  spin_data[i,1] = 1.0*np.count_nonzero(spins==-1)/(1.0*n)
  spin_data[i,2] = 1.0*np.count_nonzero(spins==0)/(1.0*n)
  spin_data[i,3] = 1.0*np.count_nonzero(spins==1)/(1.0*n)
  spin_data[i,4] = 1.0*np.count_nonzero(spins==2)/(1.0*n)

np.savetxt(mydir + 'cg_spin_pops.txt', np.c_[np.arange(0,nframe),spin_data])
