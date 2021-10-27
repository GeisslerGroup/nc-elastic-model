#include "site.h"

site::site() : nbs(6){
  
  x = 0.0;
  y = 0.0;
  s = 1;
  n = 0;
  nxn = 0;
  sublat=0;
  layer=0;
  reset_nbs();
}

void site::reset_nbs() {

  for(int i=0; i<6; i++){
    nbs(i)=-1;
  }
}

void site::set_params(double x1, double y1, int s1, int n1){

  x=x1;
  y=y1;
  s=s1;
  n=n1;
}

