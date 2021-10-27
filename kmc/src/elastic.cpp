/*** Define functions related to elasticity ***/

#include "atom.h"
#include "elastic.h"
#include <stdlib.h>
#include <math.h>
#include <armadillo>

/**************************/

double min_energy(std::vector<atom> &lat, int N, double dx, double k1, double k2, double l1, double l2){

  double ecurr=0, elast=1e10;
  int t=0;
  double tol = 1e-6;

	//for some # of steps or until tolerance is met, repeat
  elast = elastic_energy_tot(lat, N, k1, k2, l1, l2);
  while(fabs((elast-ecurr))>tol){
    elast = elastic_energy_tot(lat, N, k1, k2, l1, l2);
    t++;
		std::vector<arma::vec::fixed<3>> force(N);
		force = get_force(lat, N, k1, k2, l1, l2);
//    printf("current energy: %f\n", elast);
//    std::cout << "force:" << force[0](0) << "\n";
		for(int i=0; i<N; i++){
      //if(i!=center_index){
      lat[i].pos(0) = lat[i].pos(0) + dx*force[i](0);
      lat[i].pos(1) = lat[i].pos(1) + dx*force[i](1);
      lat[i].pos(2) = lat[i].pos(2) + dx*force[i](2);
      //}
		}
    ecurr = elastic_energy_tot(lat, N, k1, k2, l1, l2);

	}

	return ecurr;	
}

/********************************************/

double elastic_energy_tot(std::vector<atom> &lat, int N, double k1, double k2, double l1, double l2){

  double energy = 0;
  for(int i=0; i<(N-1); i++){
    for(int j=0; j<lat[i].n; j++){
      int nind = (int)lat[i].nbs[j];
      if(nind>i){
        double r = get_dist(lat[i], lat[nind]);
        double kcurr = get_k(lat[i].s, lat[nind].s, k1, k2);
        double lcurr = get_l(lat[i].s, lat[nind].s, l1, l2);
        energy += 0.5*kcurr*(r-lcurr)*(r-lcurr);
      }
    }
  }
  return energy;
}

/********************************************/

std::vector<arma::vec::fixed<3> > get_force(std::vector<atom> &lat, int N, double k1, double k2, double l1, double l2){

  std::vector<arma::vec::fixed<3> > force(N);

  for(int i=0; i<N; i++){
    force[i].zeros();
    for(int j=0; j<3; j++) force[i](j) = 0;
  }

  for(int i=0; i<N; i++){
    for(int j=0; j<lat[i].n; j++){
      int nind = (int)lat[i].nbs[j];
      double r = get_dist(lat[i], lat[nind]);
      if(r==0) printf("Error: distance is zero.\n");
      arma::vec disp(3);
      disp = get_dispvec(lat[i], lat[nind]);
      double kcurr = get_k(lat[i].s, lat[nind].s, k1, k2);
      double lcurr = get_l(lat[i].s, lat[nind].s, l1, l2);
      force[i] += -kcurr*(r-lcurr)/r*disp; 
    }
  }
  return force;
}

/*********************************************/

arma::vec get_dispvec(atom si, atom sj){

	arma::vec r(3);

  r(0) = si.pos(0)-sj.pos(0);
  r(1) = si.pos(1)-sj.pos(1);
  r(2) = si.pos(2)-sj.pos(2);

	return r;
}

/*********************************************/

double get_dist(atom si, atom sj){

	double r = arma::norm(si.pos-sj.pos);

	return r;
}

/********************************************/

double get_k(int s1, int s2, double k1, double k2){
  if(s1==1 && s2==1) return k1;
  else if(s1==-1 && s2==-1) return k2;
  else return sqrt(k1*k2);
}

/********************************************/

double get_l(int s1, int s2, double l1, double l2){
  if(s1==1 && s2==1) return l1;
  else if(s1==-1 && s2==-1) return l2;
  else return (l1+l2)/2.0;
}

