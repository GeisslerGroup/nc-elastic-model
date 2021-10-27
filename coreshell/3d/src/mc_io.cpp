#include "mc_io.h"
#include <stdio.h>
#include <stdlib.h>

/*************************/
/*** Reading Functions ***/
/*************************/

void read_potential(char *inp, arma::mat &veff){

  FILE *f;
  f = fopen(inp, "r");
  printf("opened file\n");
  for(int i=0; i<Nlat; i++){
    for(int j=0; j<Nlat; j++){
      if(fscanf(f, "%lf\n", &veff(i,j))!=1) printf("Error: could not read effective potential.\n");
      //Remove q=0 mode
      //veff(i,j) += 4/Nlat;
    }
  }
  fclose(f);
  f=NULL;
}

/**************************/
/*** Printing Functions ***/
/**************************/

void print_pop(double *fraction, const char *name, int nstp){

	FILE *f;
  f = fopen(name, "w");
	for(int i=0; i<nstp; i++){
		fprintf(f, "%d %f\n", i, fraction[i]);
	}
	fclose(f);
	f=NULL;
}

void pconf(std::vector<atom> &lat, const char *name, int time){

  FILE *f;
	f = fopen(name, "w");
	fprintf(f, "%d\n", Nlat);
	fprintf(f, "%s %d\n", "time", time);

	for(int i=0; i<Nlat; i++){
		if(lat[i].s==1) fprintf(f, "Ar %f %f %f\n", lat[i].pos(0), lat[i].pos(1), lat[i].pos(2));
    else fprintf(f, "S %f %f %f\n", lat[i].pos(0), lat[i].pos(1), lat[i].pos(2));
	}
	fclose(f);
	f=NULL;
}
