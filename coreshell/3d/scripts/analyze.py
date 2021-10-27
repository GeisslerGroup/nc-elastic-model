import sys
from math import *
import numpy as np
import scipy.spatial as spatial


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
'''
elif Ns==12:
  Kvals = [4.0,5.0,6.2,5.4,5.6,5.8,6.0,6.2,6.4,6.6,7.0,7.2,7.4,7.6,7.8,7.9,8.0,8.2,8.4,8.6,8.8,9.0,9.2]
'''
#else:
#  print('Size not supported.')

Kshell = Kvals[Kshell_ind]

N = (Ns+shell_layers)**3
Ncore = (Ns-1)**3

my_dict = {'Ar':1, 'S':-1}
new_dict = {-2:'H', -1:'He', 0:'Li', 1:'Be', 2:'B'}

mydir= '/home/frechette/ionex/kmc/equil/3d/confs/%s/3d/stiffshell/kT=%f/h=%f/Ns=%d/c=%f/shell_layers=%d/Kshell=%f/' % (bc, kT, h, Ns, c, shell_layers, Kshell)

spin_data = np.zeros((nframe, Ncore))
sr_data = np.zeros((nframe, Ncore))

#Useful function suggested by StackOverflow user Markus Jarderot:
#https://stackoverflow.com/questions/5193331/is-a-point-inside-regular-hexagon

def is_in_cube(x0, y0, z0, x, y, z, a):  
  
  dx = fabs(x-x0)
  dy = fabs(y-y0)
  dz = fabs(z-z0)
  return ((dx<=a) and (dy<=a) and (dz<=a))

######################

def get_dist(x1, x2):
  dx = x1-x2
  return dx

def write_xyz(coords, spins, thedir, natom, index):

  myfile = thedir + 'cg_%05d.xyz' % index
  myheader = '%d\n%d' % (natom, index)
  myatoms = np.array([new_dict[spin] for spin in spins])
  arr = np.zeros(myatoms.size, dtype=[('var1', 'U3'), ('var2', float), ('var3', float), ('var4', float)])
  arr['var1'] = myatoms
  arr['var2'] = coords[:,0]
  arr['var3'] = coords[:,1]
  arr['var4'] = coords[:,2]

  np.savetxt(myfile, arr, fmt='%s %f %f %f', header=myheader, comments='')

######################
print('Ns: %d, shell_layers: %d' % (Ns, shell_layers))
for i in range(nframe):  
  if(i%100==0):
    print(i)
  myseed=1
  myfile = mydir + 'kmc_conf_%05d_seed=%d.xyz' % (i,myseed) 
  conf = open(myfile)

  line = conf.readline()
  n = int(line.strip())
  all_coords = np.loadtxt(myfile, skiprows=2, usecols=(1,2,3))
  all_atom_types = np.loadtxt(myfile, dtype=str, skiprows=2, usecols=(0)).tolist()
  all_spins = np.array([my_dict[atom] for atom in all_atom_types]) 

  #Extract core atoms
  core_list = []
  extra_core_list = []
  centroid = all_coords.mean(axis=0)
  for j in range(n):
    if(is_in_cube(centroid[0], centroid[1], centroid[2], all_coords[j,0], all_coords[j,1], all_coords[j,2], (Ns-1)/2.0)):
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

  #Re-order neighbors
  '''
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
  '''

  cg_spins = np.zeros(Ncore)
  for j in range(Ncore):
    sr_data[i,j] = int(nbs_sum[j])
    spin_data[i,j] = spins[j]
    if(nbs_sum[j]==-18 and spins[j]==-1):
      cg_spins[j] = -2
    if(nbs_sum[j]==18 and spins[j]==1):
      cg_spins[j] = 2
    '''
    if(nbs_sum[j]==0 and spins[j]==-1 and extra_spins[int(nbs_ord[j,0])]==-1 and extra_spins[int(nbs_ord[j,2])]==-1 and extra_spins[int(nbs_ord[j,4])]==-1):
      cg_spins[j] = -1
    if(nbs_sum[j]==0 and spins[j]==-1 and extra_spins[int(nbs_ord[j,0])]==1 and extra_spins[int(nbs_ord[j,2])]==1 and extra_spins[int(nbs_ord[j,4])]==1):
      cg_spins[j] = -1
    if(nbs_sum[j]==0 and spins[j]==1 and extra_spins[int(nbs_ord[j,0])]==-1 and extra_spins[int(nbs_ord[j,2])]==-1 and extra_spins[int(nbs_ord[j,4])]==-1):
      cg_spins[j] = 1
    if(nbs_sum[j]==0 and spins[j]==1 and extra_spins[int(nbs_ord[j,0])]==1 and extra_spins[int(nbs_ord[j,2])]==1 and extra_spins[int(nbs_ord[j,4])]==1):
      cg_spins[j] = 1
    '''
  #Write coarse-grained configuration to file
  write_xyz(coords, cg_spins, mydir, Ncore, i)

