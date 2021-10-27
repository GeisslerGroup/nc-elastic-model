#ifndef LATTICE_H
#define LATTICE_H

#include <stdlib.h>
#include <gsl/gsl_rng.h>
#include "atom.h"

extern int nx;
extern int ny;
extern int Ns;
extern int Nlat;
extern int nlayer;
extern int shell_layers;
extern double Lx;
extern double Ly;
extern double net_c;
extern gsl_rng *rg;

void initialize(std::vector<atom> &lat, char *lat_type, char *bc, char *config);
void init_bulk(std::vector<atom> &lat, char *lat_type);
void init_strip(std::vector<atom> &lat, char *lat_type);
void init_nc(std::vector<atom> &lat, char *lat_type);

double get_dist(atom si, atom sj, char *bc);
double get_dist_bulk(double dx, double dy);
double get_dist_strip(double dx, double dy);
double get_dist_nc(double dx, double dy);

#endif
