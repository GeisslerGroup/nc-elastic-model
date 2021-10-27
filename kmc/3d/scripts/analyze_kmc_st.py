import sys
from math import *
import numpy as np
import scipy.spatial as spatial

nframe = 50

Ns = 12
kT = 1.5
h = -100.0
kswap=1.0

Ns = int(sys.argv[1])
seed=int(sys.argv[2])

N = Ns**3
Ncore = (Ns-2)**3

nframe = N

my_dict = {'Ar':1, 'S':-1}
new_dict = {-2:'H', -1:'He', 0:'Li', 1:'Be', 2:'B'}

mydir= '/home/frechette/ionex/kmc/3d/data/J=0.000000/nx=%d_ny=%d_nz=%d/kT=%f/h=%f/k_swap=%f/confs/seed=%03d/' % (Ns, Ns, Ns, kT, h, kswap, seed)

spin_data = np.zeros((nframe, Ncore))
sr_data = np.zeros((nframe, Ncore))

def is_in_cube(x0, y0, z0, x, y, z, a):  
  
  dx = fabs(x-x0)
  dy = fabs(y-y0)
  dz = fabs(z-z0)
  return ((dx<=a) and (dy<=a) and (dz<=a))

######################

def get_dist(x1, x2):
  dx = x1-x2
  return dx

def write_xyz(coords, spins, thedir, natom, index, time):

  myfile = thedir + 'cg_st_%04d.xyz' % index
  myheader = '%d\n%.15e' % (natom, time)
  myatoms = np.array([new_dict[spin] for spin in spins])
  arr = np.zeros(myatoms.size, dtype=[('var1', 'U3'), ('var2', float), ('var3', float), ('var4', float)])
  arr['var1'] = myatoms
  arr['var2'] = coords[:,0]
  arr['var3'] = coords[:,1]
  arr['var4'] = coords[:,2]

  np.savetxt(myfile, arr, fmt='%s %f %f %f', header=myheader, comments='')

######################
print("Getting CG frames for seed %d..." % seed)
for i in range(nframe):  
  myfile = mydir + 'shortt_%04d.xyz' % i 
  conf = open(myfile)

  line = conf.readline()
  n = int(line.strip())
  line2 = conf.readline()
  t = float(line2.strip()[5:])
  all_coords = np.loadtxt(myfile, skiprows=2, usecols=(1,2,3))
  all_atom_types = np.loadtxt(myfile, dtype=str, skiprows=2, usecols=(0)).tolist()
  all_spins = np.array([my_dict[atom] for atom in all_atom_types]) 

  #Extract core atoms
  core_list = []
  extra_core_list = []
  centroid = all_coords.mean(axis=0)
  for j in range(n):
    if(is_in_cube(centroid[0], centroid[1], centroid[2], all_coords[j,0], all_coords[j,1], all_coords[j,2], (Ns-2)/2.0)):
      core_list.append(j)
    if(is_in_cube(centroid[0], centroid[1], centroid[2], all_coords[j,0], all_coords[j,1], all_coords[j,2], Ns/2.0)):
      extra_core_list.append(j)
  
  coords = all_coords[core_list,:]
  extra_coords = all_coords[extra_core_list,:]
  spins = all_spins[core_list]
  extra_spins = all_spins[extra_core_list]

  #Get nearest neighbors
  mytree = spatial.cKDTree(extra_coords)
  nbs_list = mytree.query_ball_point(coords,sqrt(2.0)+0.0001)
  self_list = mytree.query_ball_point(coords,0.1)
    
  for j in range(len(nbs_list)):
    nbs_list[j].remove(self_list[j][0])
  nbs=np.vstack(nbs_list)

  #Get sum of nn spins
  nbs_sum = np.sum(extra_spins[nbs], axis=1)
  #np.savetxt(mydir + 'sr_%04d.txt' % i, nbs_sum)

  cg_spins = np.zeros(Ncore)
  for j in range(Ncore):
    sr_data[i,j] = int(nbs_sum[j])
    spin_data[i,j] = spins[j]
    if(nbs_sum[j]==-18 and spins[j]==-1):
      cg_spins[j] = -2
    if(nbs_sum[j]==18 and spins[j]==1):
      cg_spins[j] = 2

  #Write coarse-grained configuration to file
  write_xyz(coords, cg_spins, mydir, Ncore, i, t)

