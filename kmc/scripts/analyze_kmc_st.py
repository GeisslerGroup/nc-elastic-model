import sys
from math import *
import numpy as np
import scipy.spatial as spatial

Ns = 8
kT = 0.2
h = -10
kswap=1000000.0

Ns = int(sys.argv[1])
seed=int(sys.argv[2])

N = 3*Ns*Ns-3*Ns+1
Ncore = 3*(Ns-1)*(Ns-1)-3*(Ns-1)+1

nframe = N

my_dict = {'Ar':1, 'S':-1}
new_dict = {-2:'H', -1:'He', 0:'Li', 1:'Be', 2:'B'}

mydir= '/home/frechette/ionex/kmc/data/J=0.000000/Ns=%d/kT=%f/h=%f/k_swap=%f/confs/seed=%03d/' % (Ns, kT, h, kswap, seed)

spin_data = np.zeros((nframe, Ncore))
sr_data = np.zeros((nframe, Ncore))

#Useful function suggested by StackOverflow user Markus Jarderot:
#https://stackoverflow.com/questions/5193331/is-a-point-inside-regular-hexagon

def is_in_hexagon(x0, y0, d, x, y):  
  
  dx = fabs(x-x0)/d
  dy = fabs(y-y0)/d
  a = 0.25*sqrt(3.0)
  return ((dy<=a-1e-3) and ((a*dx + 0.25*dy) <= 0.5*a-1e-3))

######################

def get_dist(x1, x2):
  dx = x1-x2
  return dx

def write_xyz(coords, spins, thedir, natom, index, time):

  myfile = thedir + 'cg_st_%04d.xyz' % index
  coords3d = np.zeros((natom,3))
  coords3d[:,:-1] = coords
  myheader = '%d\n%.15e' % (natom, time)
  myatoms = np.array([new_dict[spin] for spin in spins])
  arr = np.zeros(myatoms.size, dtype=[('var1', 'U3'), ('var2', float), ('var3', float), ('var4', float)])
  arr['var1'] = myatoms
  arr['var2'] = coords3d[:,0]
  arr['var3'] = coords3d[:,1]
  arr['var4'] = coords3d[:,2]

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
  all_coords = np.loadtxt(myfile, skiprows=2, usecols=(1,2))
  all_atom_types = np.loadtxt(myfile, dtype=str, skiprows=2, usecols=(0)).tolist()
  all_spins = np.array([my_dict[atom] for atom in all_atom_types]) 

  #Extract core atoms
  core_list = []
  extra_core_list = []
  centroid = all_coords.mean(axis=0)
  for j in range(n):
    if(is_in_hexagon(centroid[0], centroid[1], 2.0*(Ns-1), all_coords[j,0], all_coords[j,1])):
      core_list.append(j)
    if(is_in_hexagon(centroid[0], centroid[1], 2.0*(Ns), all_coords[j,0], all_coords[j,1])):
      extra_core_list.append(j)
  
  coords = all_coords[core_list,:]
  extra_coords = all_coords[extra_core_list,:]
  spins = all_spins[core_list]
  extra_spins = all_spins[extra_core_list]

  #Get nearest neighbors
  mytree = spatial.cKDTree(extra_coords)
  nbs_list = mytree.query_ball_point(coords,1.0001)
  self_list = mytree.query_ball_point(coords,0.1)
    
  for j in range(len(nbs_list)):
    nbs_list[j].remove(self_list[j][0])
  nbs=np.vstack(nbs_list)

  #Get sum of nn spins
  nbs_sum = np.sum(extra_spins[nbs], axis=1)
  #np.savetxt(mydir + 'sr_%04d.txt' % i, nbs_sum)

  #Re-order neighbors
  nbs_coords = extra_coords[nbs]
  nbs_ord = np.zeros(nbs.shape)
  for j in range(Ncore):
    #print('check')
    for k in range(6):
      if(fabs(nbs_coords[j,k,1]-coords[j,1])<1e-5):
        if(get_dist(coords[j,0],nbs_coords[j,k,0])>0):
          nbs_ord[j,0] = int(nbs[j,k])
          #print(0)
        else:
          nbs_ord[j,3] = int(nbs[j,k])
          #print(3)
      if(get_dist(coords[j,1],nbs_coords[j,k,1])>0):
        if(get_dist(coords[j,0],nbs_coords[j,k,0])>0):
          nbs_ord[j,1] = int(nbs[j,k])
          #print(5)
        else:
          nbs_ord[j,2] = int(nbs[j,k])
          #print(4)
      if(get_dist(coords[j,1],nbs_coords[j,k,1])<0):
        if(get_dist(coords[j,0],nbs_coords[j,k,0])>0):
          nbs_ord[j,5] = int(nbs[j,k])
          #print(1)
        else:
          nbs_ord[j,4] = int(nbs[j,k])
          #print(2)
     

  cg_spins = np.zeros(Ncore)
  for j in range(Ncore):
    sr_data[i,j] = int(nbs_sum[j])
    spin_data[i,j] = spins[j]
    if(nbs_sum[j]==-6 and spins[j]==-1):
      cg_spins[j] = -2
    if(nbs_sum[j]==-6 and spins[j]==1):
      cg_spins[j] = -1
    if(nbs_sum[j]==6 and spins[j]==1):
      cg_spins[j] = 2
    if(nbs_sum[j]==6 and spins[j]==-1):
      cg_spins[j] = 1
    if(nbs_sum[j]==0 and spins[j]==-1 and extra_spins[int(nbs_ord[j,0])]==-1 and extra_spins[int(nbs_ord[j,2])]==-1 and extra_spins[int(nbs_ord[j,4])]==-1):
      cg_spins[j] = -1
    if(nbs_sum[j]==0 and spins[j]==-1 and extra_spins[int(nbs_ord[j,0])]==1 and extra_spins[int(nbs_ord[j,2])]==1 and extra_spins[int(nbs_ord[j,4])]==1):
      cg_spins[j] = -1
    if(nbs_sum[j]==0 and spins[j]==1 and extra_spins[int(nbs_ord[j,0])]==-1 and extra_spins[int(nbs_ord[j,2])]==-1 and extra_spins[int(nbs_ord[j,4])]==-1):
      cg_spins[j] = 1
    if(nbs_sum[j]==0 and spins[j]==1 and extra_spins[int(nbs_ord[j,0])]==1 and extra_spins[int(nbs_ord[j,2])]==1 and extra_spins[int(nbs_ord[j,4])]==1):
      cg_spins[j] = 1

  #Write coarse-grained configuration to file
  write_xyz(coords, cg_spins, mydir, Ncore, i, t)

