#ifndef SITE_H
#define SITE_H

#include <armadillo>

class site{
  
  public:
    site();

    double x;
    double y;
    int s;
    int n;
    int nxn;
    int sublat;
    int layer; //relevant for hexagonal NC

    arma::vec nbs;
    arma::vec nxt_nbs = arma::vec(6);

    void reset_nbs();
    void set_params(double x1, double y1, int s1, int n1);

    ~site(){};
};

#endif
