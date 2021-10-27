import sys
from math import *
import numpy as np
import scipy.spatial as spatial

nframe = 100

bc = 'nc'
Ns = 8
kT = 0.2
h = -10
kswap=1.0

Ns = int(sys.argv[1])
nframe = int(sys.argv[2])
seed=int(sys.argv[3])

N=3*Ns*Ns-3*Ns+1
#nframe = nframe*N

my_dict = {'Ar':1, 'S':-1}
new_dict = {-2:'H', -1:'He', 0:'Li', 1:'Be', 2:'B'}

mydir= '/home/frechette/ionex/kmc/data/J=0.000000/Ns=%d/kT=%f/h=%f/k_swap=%f/confs/seed=%03d/' % (Ns, kT, h, kswap, seed)
mypopdir= '/home/frechette/ionex/kmc/data/J=0.000000/Ns=%d/kT=%f/h=%f/k_swap=%f/pops/' % (Ns, kT, h, kswap)

times = np.zeros(nframe)
rmsvol = np.zeros(nframe)

######################

def is_in_hexagon(x0, y0, d, x, y):  
  
  dx = fabs(x-x0)/d
  dy = fabs(y-y0)/d
  a = 0.25*sqrt(3.0)
  return ((dy<=a-1e-3) and ((a*dx + 0.25*dy) <= 0.5*a-1e-3))

def get_dist(x1, x2):
  dx = x1-x2
  return dx

######################
for i in range(nframe):  
  myfile = mydir + 'conf_%04d.xyz' % i 
  dispfile = mydir + 'disp_%04d.xyz' % i
  conf = open(myfile)

  line = conf.readline()
  n = int(line.strip())
  line2 = conf.readline()
  t = float(line2.strip()[5:])
  times[i] = t
  coords = np.loadtxt(myfile, skiprows=2, usecols=(1,2))
  disp = np.loadtxt(dispfile, skiprows=2, usecols=(1,2))
  vols = np.zeros(n)

  #Extract core atoms
  core_list = []
  centroid = coords.mean(axis=0)
  for j in range(n):
    if(is_in_hexagon(centroid[0], centroid[1], 2.0*(Ns-1), coords[j,0], coords[j,1])):
      core_list.append(j)

  #Get nearest neighbors
  mytree = spatial.cKDTree(coords)
  nbs_list = mytree.query_ball_point(coords,1.0001)
  self_list = mytree.query_ball_point(coords,0.1)
    
  for j in range(len(nbs_list)):
    nbs_list[j].remove(self_list[j][0])

  for j in range(n):
    for k in range(len(nbs_list[j])):
      x1 = disp[j,0]
      x2 = disp[nbs_list[j][k],0]
      y1 = disp[j,1]
      y2 = disp[nbs_list[j][k],1]
      dist = sqrt(get_dist(x1,x2)**2+get_dist(y1,y2)**2)
      vols[j] = vols[j] + dist**2

    vols[j] = vols[j]/len(nbs_list[j])-1.0

  rmsvol[i] = sqrt(np.average(vols[core_list]**2))

  np.savetxt(mydir + 'vols_%04d.txt' % i, np.c_[coords, vols])
  np.savetxt(mydir + 'vols_core_%04d.txt' % i, np.c_[coords[core_list], vols[core_list]])

np.savetxt(mypopdir + 'rmsvol_seed=%d.txt' % seed, np.c_[times, rmsvol])
