import sys
from math import *
import numpy as np
import scipy.spatial as spatial

bc = 'nc'
Ns = 12
kT = 1.5
h = -100
kswap=1.0

Ns = int(sys.argv[1])
seed=int(sys.argv[2])

N=Ns**3

nframe=N

my_dict = {'Ar':1, 'S':-1}
new_dict = {-2:'H', -1:'He', 0:'Li', 1:'Be', 2:'B'}

mydir= '/home/frechette/ionex/kmc/3d/data/J=0.000000/nx=%d_ny=%d_nz=%d/kT=%f/h=%f/k_swap=%f/confs/seed=%03d/' % (Ns, Ns, Ns, kT, h, kswap, seed)
mypopdir= '/home/frechette/ionex/kmc/3d/data/J=0.000000/nx=%d_ny=%d_nz=%d/kT=%f/h=%f/k_swap=%f/pops/' % (Ns, Ns, Ns, kT, h, kswap)

times = np.zeros(nframe)
rmsvol = np.zeros(nframe)

######################

def is_in_cube(x0, y0, z0, x, y, z, a):  
  
  dx = fabs(x-x0)
  dy = fabs(y-y0)
  dz = fabs(z-z0)
  return ((dx<=a) and (dy<=a) and (dz<=a))

def get_dist(x1, x2):
  dx = x1-x2
  return dx

######################
for i in range(nframe):  
  myfile = mydir + 'shortt_%04d.xyz' % i 
  dispfile = mydir + 'disp_st_%04d.xyz' % i
  conf = open(myfile)

  line = conf.readline()
  n = int(line.strip())
  line2 = conf.readline()
  t = float(line2.strip()[5:])
  times[i] = t
  coords = np.loadtxt(myfile, skiprows=2, usecols=(1,2,3))
  disp = np.loadtxt(dispfile, skiprows=2, usecols=(1,2,3))
  vols = np.zeros(n)

  #Extract core atoms
  core_list = []
  centroid = coords.mean(axis=0)
  for j in range(n):
    if(is_in_cube(centroid[0], centroid[1], centroid[2], coords[j,0], coords[j,1], coords[j,2], (Ns-2)/2.0)):
      core_list.append(j)

  #Get nearest neighbors
  mytree = spatial.cKDTree(coords)
  nbs_list = mytree.query_ball_point(coords,1.0+0.0001)
  self_list = mytree.query_ball_point(coords,0.1)
    
  for j in range(len(nbs_list)):
    nbs_list[j].remove(self_list[j][0])

  for j in range(n):
    for k in range(len(nbs_list[j])):
      x1 = disp[j,0]
      x2 = disp[nbs_list[j][k],0]
      y1 = disp[j,1]
      y2 = disp[nbs_list[j][k],1]
      z1 = disp[j,2]
      z2 = disp[nbs_list[j][k],2]
      dist = sqrt(get_dist(x1,x2)**2+get_dist(y1,y2)**2+get_dist(z1,z2)**2)
      if(dist>10.0):
        print("WHOA, SOMETHING'S WRONG: DIST=%f" % dist)
      vols[j] = vols[j] + dist**3

    vols[j] = vols[j]/len(nbs_list[j])-1.0

  rmsvol[i] = sqrt(np.average(vols[core_list]**2))

  np.savetxt(mydir + 'vols_st_%04d.txt' % i, np.c_[coords, vols])

np.savetxt(mypopdir + 'rmsvol_st_seed=%d.txt' % seed, np.c_[times, rmsvol])
