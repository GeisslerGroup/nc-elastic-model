#ifndef MC_H
#define MC_H

#include "atom.h"
#include <stdlib.h>
#include <armadillo>

extern int Nlat;
extern double J;
extern double h;
extern double kT;

double run_mc_equil(std::vector<atom> &lat, int nstp, arma::mat &veff, char *dynamics);
double run_mc_traj(std::vector<atom> &lat, double *fraction, int nstp, arma::mat &veff, char *dynamics, char *confdir);

//Energy Functions
double get_field_energy(std::vector<atom> &lat);
double get_pair_energy(std::vector<atom> &lat);
double get_elastic_energy(std::vector<atom> &lat, arma::mat &veff);
double get_elastic_flip(std::vector<atom> &lat, arma::mat &veff, int j);
double get_elastic_swap(std::vector<atom> &lat, arma::mat &veff, int j, int k);
double get_swap_energy(std::vector<atom> &lat, int index1, int index2);
double get_boltzmann_flip(std::vector<atom> &lat, int ind);
double get_boltzmann_swap(std::vector<atom> &lat, int ind1, int ind2);
double get_boltzmann_flip_elastic(std::vector<atom> &lat, int ind, arma::mat &v);
double get_boltzmann_swap_elastic(std::vector<atom> &lat, int ind1, int ind2, arma::mat &v);

double get_nup(std::vector<atom> &lat);

#endif
