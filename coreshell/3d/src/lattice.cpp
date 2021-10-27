/*********************************************************
 * This file contains functions for initializing lattices
 * and computing distances on lattices with different
 * boundary conditions.
 ********************************************************/

#include <stdlib.h>
#include <gsl/gsl_rng.h>
#include "lattice.h"
#include "atom.h"

/*********************************************************/

void initialize(std::vector<atom> &lat, char *lat_type, char *bc, char *config){

  if(strcmp(config, "phase_sep")==0){
    for(int i=0; i<Nlat; i++){
      if(i<(int)(Nlat*net_c)) lat[i].s = 1;
      else lat[i].s = -1;
    }
  }
  else if(strcmp(config, "random")==0){
    for(int i=0; i<Nlat; i++) lat[i].s = -1;
    for(int i=0; i<(int)(Nlat*net_c); i++){
      int found = 0;
      while(found==0){
        int index = (int)gsl_rng_uniform_int(rg, Nlat);
        if(lat[index].s==-1){
          found = 1;
          lat[index].s = 1;
        }
      }
    }
  }
  else{
    printf("Initial configuration not supported by \"lattice.cpp\".\n");
    exit(-1);
  }

  if(strcmp(bc, "bulk")==0){
    init_bulk(lat, lat_type);
  }
  else if(strcmp(bc, "strip")==0){
    init_strip(lat, lat_type);
  }
  else if(strcmp(bc, "nc")==0){
    init_nc(lat, lat_type);
  }
  else{
    printf("Boundary condition not supported by \"lattice.cpp\".\n");
    exit(-1);
  } 
}

void init_bulk(std::vector<atom> &lat, char *lat_type){

  if(strcmp(lat_type, "cube")==0){
    Lx = 1.0*nx;
    Ly = 1.0*ny;
    Lz = 1.0*nz;
    for(int i=0; i<nx; i++){
      for(int j=0; j<ny; j++){
        for(int k=0; k<nz; k++){
          lat[i+j*nx+k*nx*ny].pos(0) = 1.0*i;
          lat[i+j*nx+k*nx*ny].pos(1) = 1.0*j;
          lat[i+j*nx+k*nx*ny].pos(2) = 1.0*k;
        }
      }
    }
  }
  else{
    printf("Lattice geometry not supported by \"lattice.cpp\".\n");
    exit(-1);
  }

  //Get nearest neighbors
  for(int i=0; i<Nlat; i++){
    int nn=0;
    for(int j=0; j<Nlat; j++){
      if(i!=j){
        if(get_dist(lat[i], lat[j], strdup("bulk"))<1.0001){
          lat[i].nbs.push_back(j);
          nn++;
        }
      }
    }
    lat[i].n = nn;
  }
}

void init_strip(std::vector<atom> &lat, char *lat_type){

  if(strcmp(lat_type, "square")==0){
    Lx = 1.0*nx;
    Ly = 1.0*ny;
    for(int i=0; i<nx; i++){
      for(int j=0; j<ny; j++){
       lat[i+j*nx].pos(0) = 1.0*i;
       lat[i+j*nx].pos(1) = 1.0*j;
      }
    }
  }
  else if(strcmp(lat_type, "triangular")==0){
    Lx = 1.0*nx;
    Ly = sqrt(3.0)/2.0*(ny+2*nlayer);
    for(int i=0; i<nx; i++){
      for(int j=0; j<(ny+2*nlayer); j++){
       lat[i+j*nx].pos(0) = 1.0*i + 0.5*((j+1)%2);
       lat[i+j*nx].pos(1) = sqrt(3.0)*j/2.0;
      }
    }
  }
  else{
    printf("Lattice geometry not supported by \"lattice.cpp\".\n");
    exit(-1);
  }

  for(int i=0; i<Nlat; i++){
    if(i<nlayer*nx) lat[i].s=-1;
    if(i>(Nlat-nlayer*nx-1)) lat[i].s=-1;
  }

  //Get nearest neighbors
  for(int i=0; i<Nlat; i++){
    int nn=0;
    for(int j=0; j<Nlat; j++){
      if(i!=j){
        if(get_dist(lat[i], lat[j], strdup("strip"))<1.0001){
          lat[i].nbs.push_back(j);
          nn++;
        }
      }
    }
    lat[i].n = nn;
  }
}

void init_nc(std::vector<atom> &lat, char *lat_type){

  if(strcmp(lat_type, "cube")==0){

    //int nl = Ns+shell_layers;
    int nlx = nx+shell_layers;
		int nly = ny+shell_layers;
		int nlz = nz+shell_layers;
    for(int i=0; i<nlx; i++){
      for(int j=0; j<nly; j++){
        for(int k=0; k<nlz; k++){
          lat[i+j*nlx+k*nlx*nly].layer = std::min({i+1,j+1,k+1,nlx-i,nly-j,nlz-k});
          lat[i+j*nlx+k*nlx*nly].pos(0) = 1.0*i;
          lat[i+j*nlx+k*nlx*nly].pos(1) = 1.0*j;
          lat[i+j*nlx+k*nlx*nly].pos(2) = 1.0*k;
        }
      }
    }
  }

  else if(strcmp(lat_type, "triangular")==0){

    int cnt = 0;

    double xs=0;
    double ys=0;

    for(int i=0; i<Ns+shell_layers; i++){
      xs = (-1.0/2.0)*i;
      ys = (sqrt(3.0)/2.0)*i;
      for(int j=0; j<(Ns+shell_layers+i); j++){
        lat[cnt].layer = std::min(i+1, std::min(j+1, Ns+shell_layers+i-j));
        lat[cnt].pos(0) = xs + 1.0*j;
        lat[cnt].pos(1) = ys;	
        cnt++;
      }
    }
    for(int i=(Ns+shell_layers-2); i>=0; i--){
      xs += 1.0/2.0;
      ys += sqrt(3.0)/2.0;
      for(int j=0; j<(Ns+shell_layers+i); j++){
        lat[cnt].layer = std::min(i+1, std::min(j+1, Ns+shell_layers+i-j));
        lat[cnt].pos(0) = xs + 1.0*j;
        lat[cnt].pos(1) = ys;
        cnt++;
      }
    }
    Nlat=cnt;
    cnt = 0;
  }

  //Get nearest neighbors
  for(int i=0; i<Nlat; i++){
    int nn=0;
    for(int j=0; j<Nlat; j++){
      if(i!=j){
        if(get_dist(lat[i], lat[j], strdup("nc"))<1.0001){
          lat[i].nbs.push_back(j);
          nn++;
        }
      }
    }
    lat[i].n = nn;
  }

  //Set composition of shell
  for(int i=0; i<Nlat; i++){
    if(lat[i].layer<=shell_layers) lat[i].s = -1;
  }
}

/*********************************************************/

double get_dist(atom si, atom sj, char *bc){

  double dx = si.pos(0)-sj.pos(0);
  double dy = si.pos(1)-sj.pos(1);
  double dz = si.pos(2)-sj.pos(2);

  if(strcmp(bc, "bulk")==0) return get_dist_bulk(dx, dy, dz);
  else if(strcmp(bc, "strip")==0) return get_dist_strip(dx, dy);
  else if(strcmp(bc, "nc")==0) return get_dist_nc(dx, dy, dz);
  else{
    printf("Boundary condition not supported by \"lattice.cpp\".\n");
    exit(-1);
  }
}

double get_dist_bulk(double dx, double dy, double dz){

	if(dx>Lx*0.5) dx=dx-Lx;
	if(dx<=-Lx*0.5) dx=dx+Lx;
	if(dy>Ly*0.5) dy=dy-Ly;
	if(dy<=-Ly*0.5) dy=dy+Ly;
  if(dz>Lz*0.5) dz=dz-Lz;
  if(dz<=-Lz*0.5) dz=dz+Lz;

  return sqrt(dx*dx + dy*dy + dz*dz);
}

double get_dist_strip(double dx, double dy){

	if(dx>Lx*0.5) dx=dx-Lx;
	if(dx<=-Lx*0.5) dx=dx+Lx;

  return sqrt(dx*dx + dy*dy);
}

double get_dist_nc(double dx, double dy, double dz){

  return sqrt(dx*dx + dy*dy + dz*dz);
}
