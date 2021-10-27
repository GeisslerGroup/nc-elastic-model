#ifndef KMC_H
#define KMC_H

#include "atom.h"
#include <armadillo>

void init_rng(long unsigned int seed); //move to another header

void initialize(std::vector<atom> &lat, int *surf_ind, const char *init);
double run_traj(std::vector<atom> &lat, double *fraction, double *time, int *surf_ind, std::vector< std::pair <int, int> > &pair_ind, int nstp);
double run_traj_elastic(std::vector<atom> &lat, double *fraction, double *time, int *surf_ind, std::vector< std::pair <int, int> > &pair_ind, int nstp, arma::mat &v);

int get_pair_indices(std::vector<atom> &lat, std::vector< std::pair <int, int> > &pair_ind);

void read_potential(char *inp, arma::mat &veff);

//Energy Functions
double get_nup(std::vector<atom> &lat);
double get_field_energy(std::vector<atom> &lat);
double get_pair_energy(std::vector<atom> &lat);
double get_elastic_energy(std::vector<atom> &lat, arma::mat &veff);
double get_swap_energy(std::vector<atom> &lat, int index1, int index2);
double get_boltzmann_flip(std::vector<atom> &lat, int ind);
double get_boltzmann_swap(std::vector<atom> &lat, int ind1, int ind2);
double get_boltzmann_flip_elastic(std::vector<atom> &lat, int ind, arma::mat &v);
double get_boltzmann_swap_elastic(std::vector<atom> &lat, int ind1, int ind2, arma::mat &v);


//Printing
void print_pop(double *time, double *fraction, int nstp);
void pconf(std::vector<atom> &lat, const char *name, double time);

#endif
