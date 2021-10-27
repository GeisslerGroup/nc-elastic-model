#ifndef ELASTIC_H
#define ELASTIC_H

#include "atom.h"
#include <armadillo>

/*** Define functions related to elasticity ***/

double min_energy(std::vector<atom> &lat, int N, double dx, double k1, double k2, double l1, double l2);
double elastic_energy_tot(std::vector<atom> &lat, int N, double k1, double k2, double l1, double l2);
std::vector<arma::vec::fixed<3>> get_force(std::vector<atom> &lat, int N, double k1, double k2, double l1, double l2);
arma::vec get_dispvec(atom si, atom sj);
double get_dist(atom si, atom sj);

double get_k(int s1, int s2, double k1, double k2);
double get_l(int s1, int s2, double l1, double l2);

#endif
