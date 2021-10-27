#ifndef ELASTIC_ISING_H
#define ELASTIC_ISING_H

#include "site.h"
#include <armadillo>

void print_config(int **r, int num);
void initialize(std::vector<site> &lat, char *init);
void mc_pass(std::vector<site> &lat, double beta, const char *dynamics);
int accept_disp(std::vector<site> &lat, double beta, int m1);
int accept_spin(std::vector<site> &lat, double beta, int m1, const char *dynamics);
double ising_energy_metro(std::vector<site> &lat, int m);
double elastic_energy_metro(std::vector<site> &lat, int m);
double ising_energy_tot(std::vector<site> &lat);
double elastic_energy_tot(std::vector<site> &lat);
double get_dist(site si, site sj);
double get_k(int s1, int s2);
double get_l(int s1, int s2);
double get_bavg(std::vector<site> &lat);
site get_com(std::vector<site> &lat);
double magnetization(std::vector<site> &lat);
void pxyz(std::vector<site> &lat, int N, const char *name);

void fill_autocorr_single(double *autocorr_data, double *z_data);
void fill_spin_autocorr(double **spin_autocorr_data, double **all_spin_data, std::vector<site> &lat);

void pautocorr_single(double *autocorr_data, char *outp);
void pautocorr_mult(double **autocorr_data, char *outp);
void pquant(double *data, const char *name);

int tryswap(int t1, int t2, std::vector<std::vector<site>> &lat, double *temperatures, double *temps_orig);
void exchange(int *parity, int *tot_num, int *num_swap, std::vector<std::vector<site>> &lat, double *temperatures, double *temps_orig);

void print_datahist(double **energy_history, const char *outp);
void print_data(double *data, double *temps, const char *outp);

double elastic_energy_kawa(std::vector<site> &lat, int m0, int m1);
int is_neighbor(std::vector<site> &lat, int m1, int m2);

#endif
