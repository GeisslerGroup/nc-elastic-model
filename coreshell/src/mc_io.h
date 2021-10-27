#ifndef MC_IO_H
#define MC_IO_H

#include <stdlib.h>
#include <armadillo>
#include "atom.h"

extern int Nlat;

void read_potential(char *inp, arma::mat &veff);
void print_pop(double *fraction, const char *name, int nstp);
void pconf(std::vector<atom> &lat, const char *name, int time);

#endif
