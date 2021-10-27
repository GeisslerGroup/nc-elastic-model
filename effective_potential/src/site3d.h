#ifndef SITE3D_H
#define SITE3D_H

#include <armadillo>

class site3d{
  
  public:
    site3d();

    double x;
    double y;
    double z;
    int s;
    int n;
    int sublat;
    int layer;

    arma::vec nbs;

    void reset_nbs();
    void set_params(double x1, double y1, double z1, int s1, int n1);

    ~site3d(){};
};

#endif
