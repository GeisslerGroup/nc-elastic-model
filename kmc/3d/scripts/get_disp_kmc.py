import sys
from math import *
import numpy as np
import numpy.linalg as LA
import scipy.spatial as spatial

nframe = 50

bc = 'nc'
shape='cube'
kT = 1.5
h = -100.0
kswap = 1.0

Ns = int(sys.argv[1])
nframe = int(sys.argv[2])
seed = int(sys.argv[3])

N=Ns**3

delta = 0.1

my_dict = {'Ar':1, 'S':-1}
inv_dict = {1:'Ar', -1:'S'}

mat_dir = '../../nc_matrices/%s/' % shape
dat_dir = '/home/frechette/ionex/kmc/3d/data/J=0.000000/nx=%d_ny=%d_nz=%d/kT=%f/h=%f/k_swap=%f/confs/seed=%03d/' % (Ns, Ns, Ns, kT, h, kswap, seed)

mat_prod = np.loadtxt(mat_dir + 'conv_mat_nx=%d_ny=%d_nz=%d.txt' % (Ns,Ns,Ns))

#Compute matrix for calculating displacement field
mat_prod = np.concatenate((np.zeros(mat_prod.shape[1])[np.newaxis,:], mat_prod), axis=0)
mat_prod = np.concatenate((np.zeros(mat_prod.shape[1])[np.newaxis,:], mat_prod), axis=0)
mat_prod = np.concatenate((np.zeros(mat_prod.shape[1])[np.newaxis,:], mat_prod), axis=0)
mat_prod = np.insert(mat_prod, 3*(Ns*(Ns-1)+1)-3, np.zeros(mat_prod.shape[1]), axis=0)
mat_prod = np.insert(mat_prod, 3*Ns*Ns-1, np.zeros(mat_prod.shape[1]), axis=0)
mat_prod = np.insert(mat_prod, 3*Ns*Ns*Ns-2, np.zeros(mat_prod.shape[1]), axis=0)

#################################

def write_xyz(coords, spins, thedir, natom, index):

  myfile = thedir + 'disp_%04d.xyz' % index
  myheader = '%d\n%d' % (natom, index)
  myatoms = np.array([inv_dict[int(round(spin))] for spin in spins])
  arr = np.zeros(myatoms.size, dtype=[('var1', 'U3'), ('var2', float), ('var3', float), ('var4', float)])
  arr['var1'] = myatoms
  arr['var2'] = coords[:,0]
  arr['var3'] = coords[:,1]
  arr['var4'] = coords[:,2]

  np.savetxt(myfile, arr, fmt='%s %f %f %f', header=myheader, comments='')

#################################
#MAIN

N = Ns**3
print('seed: %d' % seed)
for i in range(0, nframe):  

  print(i)

  myfile = dat_dir + 'conf_%04d.xyz' % i 
  conf = open(myfile)
  line = conf.readline()
  n = int(line.strip())
  coords = np.loadtxt(myfile, skiprows=2, usecols=(1,2,3))
  atom_types = np.loadtxt(myfile, dtype=str, skiprows=2, usecols=(0)).tolist()
  spins = np.array([my_dict[atom] for atom in atom_types]) 
  savg = np.average(spins)
  spins = spins - savg
  disps = 0.5*np.matmul(mat_prod, spins)
  disps = disps.reshape((N, 3))
  
  write_xyz(coords+delta*disps, spins+savg, dat_dir, N, i)
