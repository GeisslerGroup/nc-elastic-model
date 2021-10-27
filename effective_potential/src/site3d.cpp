#include "site3d.h"

site3d::site3d() : nbs(18) {
  
  x = 0.0;
  y = 0.0;
  z = 0.0;
  s = 1;
  n = 0;
  layer=0;
  reset_nbs();
}

void site3d::reset_nbs() {

  for(int i=0; i<18; i++){
    nbs(i)=-1;
  }
}

void site3d::set_params(double x1, double y1, double z1, int s1, int n1){

  x=x1;
  y=y1;
  z=z1;
  s=s1;
  n=n1;
}

