import sys
from math import *
import numpy as np
import numpy.linalg as LA
import scipy.spatial as spatial

bc = 'nc'
shape='hexagon'
kT = 0.2
h = -10.0
kswap = 1000000.0

Ns = int(sys.argv[1])
seed = int(sys.argv[2])

N=3*Ns*Ns-3*Ns+1
nframe = N

delta = 0.1

my_dict = {'Ar':1, 'S':-1}
inv_dict = {1:'Ar', -1:'S'}

mat_dir = '../nc_matrices/%s/' % shape
dat_dir = '/home/frechette/ionex/kmc/data/J=0.000000/Ns=%d/kT=%f/h=%f/k_swap=%f/confs/seed=%03d/' % (Ns, kT, h, kswap, seed)

dyn_mat = np.loadtxt(mat_dir + 'dyn_mat_Ns=%d.txt' % Ns)
coup_mat = np.loadtxt(mat_dir + 'coup_mat_Ns=%d.txt' % Ns)

#Compute matrix for calculating displacement field
print(dyn_mat.shape)
dinv = LA.inv(dyn_mat)
mat_prod = np.matmul(dinv, coup_mat)
mat_prod = np.concatenate((np.zeros(mat_prod.shape[1])[np.newaxis,:], mat_prod), axis=0)
mat_prod = np.concatenate((np.zeros(mat_prod.shape[1])[np.newaxis,:], mat_prod), axis=0)
mat_prod = np.insert(mat_prod, 2*Ns-1, np.zeros(mat_prod.shape[1]), axis=0)
print(mat_prod.shape)

#################################

def write_xyz(coords, spins, thedir, natom, index):

  myfile = thedir + 'disp_st_%04d.xyz' % index
  coords3d = np.zeros((natom,3))
  coords3d[:,:-1] = coords
  myheader = '%d\n%d' % (natom, index)
  myatoms = np.array([inv_dict[int(round(spin))] for spin in spins])
  arr = np.zeros(myatoms.size, dtype=[('var1', 'U3'), ('var2', float), ('var3', float), ('var4', float)])
  arr['var1'] = myatoms
  arr['var2'] = coords3d[:,0]
  arr['var3'] = coords3d[:,1]
  arr['var4'] = coords3d[:,2]

  np.savetxt(myfile, arr, fmt='%s %f %f %f', header=myheader, comments='')

def write_disp(disps, thedir, index):

  myfile = thedir + 'disp_st_%04d.txt' % index

  np.savetxt(myfile, disps)

#################################
#MAIN

N = 3*Ns*Ns-3*Ns+1
print('seed: %d' % seed)
for i in range(0, nframe):  

  #print(i)

  myfile = dat_dir + 'shortt_%04d.xyz' % i 
  conf = open(myfile)
  line = conf.readline()
  n = int(line.strip())
  coords = np.loadtxt(myfile, skiprows=2, usecols=(1,2))
  atom_types = np.loadtxt(myfile, dtype=str, skiprows=2, usecols=(0)).tolist()
  spins = np.array([my_dict[atom] for atom in atom_types]) 
  savg = np.average(spins)
  spins = spins - savg
  disps = 0.5*np.matmul(mat_prod, spins)
  disps = disps.reshape((N, 2))
  
  write_xyz(coords+delta*disps, spins+savg, dat_dir, N, i)
  #write_disp(disps, dat_dir, i)
