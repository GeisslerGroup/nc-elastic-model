//Perform an equilibrium MC simulation of an elastic nanocrystal

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_rng.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <armadillo>

//User-defined headers
#include "mc.h"
#include "atom.h"
#include "lattice.h"
#include "mc_io.h"
#include "io.h"
#include "rand.h"

//Global Variables
gsl_rng *rg; //Random number generation

int nsteps = 100;
int nequil = 100;
int print_freq;

int Nlat;
int Ncore;
int nx = 12;
int ny = 14;
int nlayer;
int shell_layers;
int Ns;

double Lx;
double Ly;

double h = 0.0;
double J = 0.0;
double kT = 4.0;
double net_c;

long unsigned int myseed = 1;

double run_mc_equil(std::vector<atom> &lat, int nstp, arma::mat &veff);
double run_mc_traj(std::vector<atom> &lat, double *fraction, int nstp,
                arma::mat &veff);

/*** Main ***/

int main(int argc, char *argv[]){

  if(argc != 8){
    printf("Error: require seven arguments (field, temperature, nsteps, net_c, Ns, shell_layers, seed.)\n");
    exit(-1);
  }
  else{
    h = atof(argv[1]);
    kT = atof(argv[2]);
    nsteps = atoi(argv[3]);
    net_c = atof(argv[4]);
    Ns = atoi(argv[5]);
    shell_layers = atoi(argv[6]);
    myseed = atoi(argv[7]);
  }


	//Compute global variables  
  char lat_type[100] = "triangular";
  int Next = Ns+shell_layers;
  if(strcmp(lat_type, "triangular")==0){
    Nlat = 3*Next*Next-3*Next+1;
    Ncore = 3*Ns*Ns-3*Ns+1;
  }
  else if(strcmp(lat_type, "square")==0){
    Nlat = Next*Next;
    Ncore = Ns*Ns;
  }
  print_freq = 1;

	//Set up random number generator
	init_rng(myseed);

	//Memory allocation
	std::vector<atom> lat(Nlat);
  arma::mat veff(Nlat, Nlat, arma::fill::zeros);
	double *fraction = (double*)calloc((size_t)nsteps, sizeof(double));
	
	//Set up 
  char bc[100] = "nc";
  char config[100] = "phase_sep";
  char dynamics[100] = "kawasaki";
	initialize(lat, lat_type, bc, config);

  //Get effective potential
  printf("Reading effective potential...\n");
  char input[1000];
  sprintf(input, "/home/frechette/ionex/nc_matrices/veff_vec_Ns=%d.txt", Ns+shell_layers);
  struct stat fileStat0;
  if(stat(input, &fileStat0)!=0){
    printf("Error: effective potential file does not exist.\n");
    exit(-1);
  }
  read_potential(input, veff);
  printf("Read in effective potential.\n");
	
  //Equilibrating
  printf("Equilibrating...\n");
  run_mc_equil(lat, nequil, veff, dynamics);

	//Run MC
  char confdir[200];
  sprintf(confdir, "confs/nc/kT=%f/h=%f/Ns=%d/c=%f/shell_layers=%d/", kT, h, Ns, net_c, shell_layers);
  struct stat fileStat;
  if(stat(confdir, &fileStat)==-1){
    _mkdir(confdir);
    printf("Created directory.\n");
  }

  printf("Running trajectory with seed %lu.\n", myseed);
  run_mc_traj(lat, fraction, nsteps, veff, dynamics, confdir);

  //Print Data
  
  //Check if directory exists
  char dir[200];
  sprintf(dir, "pops/nc/kT=%f/h=%f/Ns=%d/c=%f/shell_layers=%d/", kT, h, Ns, net_c, shell_layers);
  struct stat fileStat2;
  if(stat(dir, &fileStat2)==-1){
    _mkdir(dir);
    printf("Created directory.\n");
  }
  char outp[300];
  sprintf(outp, "%s", dir);
  sprintf(outp+strlen(outp), "kmc_pop_seed=%lu.txt", myseed);
	print_pop(fraction, outp, nsteps);

	return 1;
}

/***************************/
/*** Key Functions ***/
/***************************/

double run_mc_equil(std::vector<atom> &lat, int nstp, arma::mat &veff, char *dynamics){

	for(int t=0; t<nstp; t++){

    //Print configurations
    if(t%100==0) std::cout << "Pass number " << t << "\n";
    if(t<100){
      char outp[300];
//      sprintf(outp, "../../phase_sep/nc/confs/equil_%04d.xyz", t);
      sprintf(outp, "equil_%04d.xyz", t);
      pconf(lat, outp, t);
    }

    //Do passes
    for(int i=0; i<Ncore; i++){
      if(strcmp(dynamics, "metropolis")==0){
        unsigned long int m2 = gsl_rng_uniform_int(rg, (unsigned long int)Nlat);
        double prob = get_boltzmann_flip_elastic(lat, (int)m2, veff); 
        double xsi = gsl_rng_uniform(rg);
        if(prob>xsi) lat[m2].s = -lat[m2].s;
      }
      else if(strcmp(dynamics, "kawasaki")==0){
        unsigned long int m1 = gsl_rng_uniform_int(rg,(unsigned long int)Nlat);
        while(lat[(int)m1].layer<=shell_layers) m1 = gsl_rng_uniform_int(rg,(unsigned long int)Nlat);
        unsigned long int m2 = gsl_rng_uniform_int(rg,(unsigned long int)Nlat);
        while((int)m2==(int)m1 || lat[(int)m2].layer<=shell_layers) m2 = gsl_rng_uniform_int(rg,(unsigned long int)Nlat);
        double prob = get_boltzmann_swap_elastic(lat, (int)m1, (int)m2, veff);
        double xsi = gsl_rng_uniform(rg);
        if(prob>xsi){
          int temp = lat[m1].s;
          lat[m1].s = lat[m2].s;
          lat[m2].s = temp;
        }
      }
      else{
        printf("This dynamics is not yet supported.\n");
        exit(-1);
      }
    }
  }

  return 1;
}

double run_mc_traj(std::vector<atom> &lat, double *fraction, int nstp, arma::mat &veff, char *dynamics, char *dir){

	for(int t=0; t<nstp; t++){

    //Print configurations
    if(t%print_freq == 0){
      if(t%100==0) std::cout << "Pass number " << t << "\n";
      char outp[300];
      sprintf(outp, "%s", dir);
      sprintf(outp+strlen(outp), "kmc_conf_%04d_seed=%lu.xyz", t/print_freq, myseed);
      //printf("%s\n", outp);
      pconf(lat, outp, t);
    } 

    //Record data
    fraction[t] = (1.0*get_nup(lat))/(1.0*Nlat); 
    
    //Do passes
    for(int i=0; i<Ncore; i++){
      if(strcmp(dynamics, "metropolis")==0){
        unsigned long int m2 = gsl_rng_uniform_int(rg, (unsigned long int)Nlat);
        double prob = get_boltzmann_flip_elastic(lat, (int)m2, veff); 
        double xsi = gsl_rng_uniform(rg);
        if(prob>xsi) lat[m2].s = -lat[m2].s;
      }
      else if(strcmp(dynamics, "kawasaki")==0){
        unsigned long int m1 = gsl_rng_uniform_int(rg,(unsigned long int)Nlat);
        while(lat[(int)m1].layer<=shell_layers) m1 = gsl_rng_uniform_int(rg,(unsigned long int)Nlat);
        unsigned long int m2 = gsl_rng_uniform_int(rg,(unsigned long int)Nlat);
        while((int)m2==(int)m1 || lat[(int)m2].layer<=shell_layers) m2 = gsl_rng_uniform_int(rg,(unsigned long int)Nlat);
        double prob = get_boltzmann_swap_elastic(lat, (int)m1, (int)m2, veff);
        double xsi = gsl_rng_uniform(rg);
        if(prob>xsi){
          int temp = lat[m1].s;
          lat[m1].s = lat[m2].s;
          lat[m2].s = temp;
        }
      }  
      else{
        printf("This dynamics is not yet supported.\n");
        exit(-1);
      }
    }
	}

  return 1;
}
