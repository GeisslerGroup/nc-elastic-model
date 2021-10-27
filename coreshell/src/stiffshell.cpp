//Perform equilibrium MC simulation of an elastic nanocrystal with a stiff shell

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

int is_arr_job;
int Kval_index;
int numK;

int nsteps = 1000;
int nequil = 10000;
int print_freq;

int Nlat;
int Ncore;
int nx = 12;
int ny = 14;
int nlayer;
int shell_layers;
double Kshell;
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

  if(argc != 9){
    printf("Error: require eight arguments (is_arr_job, field, temperature, nsteps, net_c, Ns, shell_layers, seed.)\n");
    exit(-1);
  }
  else{
    is_arr_job = atoi(argv[1]);
    h = atof(argv[2]);
    kT = atof(argv[3]);
    nsteps = atoi(argv[4]);
    net_c = atof(argv[5]);
    Ns = atoi(argv[6]);
    shell_layers = atoi(argv[7]);
    myseed = atoi(argv[8]);
  }

  nequil=nsteps;

	if(is_arr_job==1){
	  char *taskID_string;
  	int taskID;

  	taskID_string = getenv("SGE_TASK_ID");
  
  	if(taskID_string==NULL){
    	exit(EXIT_FAILURE);
  	}
  	if(1 != sscanf(taskID_string, "%d", &taskID)){
    	exit(EXIT_FAILURE);
  	}
  	printf("The value of SGE_TASK_ID is: %d\n", taskID);
  
  	Kval_index = taskID-1;
  }
  else{
		Kval_index = 0;
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

  //Get value of K_shell from index
  double *Kvals;
  if(Ns==8){
    numK = 10;
    if(Kval_index>=numK){
      printf("Error: index out of range.\n");
      exit(-1);
    }
    Kvals = (double*)calloc((size_t)numK, sizeof(double));
    Kvals[0] = 1.0;
    Kvals[1] = 2.0;
    Kvals[2] = 3.0;
    Kvals[3] = 4.0;
    Kvals[4] = 5.0;
    Kvals[5] = 6.0;
    Kvals[6] = 7.0;
    Kvals[7] = 8.0;
    Kvals[8] = 9.0;
    Kvals[9] = 10.0;
  }
  else if(Ns==10){
    numK = 10;
    if(Kval_index>=numK){
      printf("Error: index out of range.\n");
      exit(-1);
    }
    Kvals = (double*)calloc((size_t)numK, sizeof(double));
    Kvals[0] = 1.0;
    Kvals[1] = 2.0;
    Kvals[2] = 3.0;
    Kvals[3] = 4.0;
    Kvals[4] = 5.0;
    Kvals[5] = 6.0;
    Kvals[6] = 7.0;
    Kvals[7] = 8.0;
    Kvals[8] = 9.0;
    Kvals[9] = 10.0;
  }
  else if(Ns==12){
    numK = 10;
    if(Kval_index>=numK){
      printf("Error: index out of range.\n");
      exit(-1);
    }
    Kvals = (double*)calloc((size_t)numK, sizeof(double));
    Kvals[0] = 1.0;
    Kvals[1] = 2.0;
    Kvals[2] = 3.0;
    Kvals[3] = 4.0;
    Kvals[4] = 5.0;
    Kvals[5] = 6.0;
    Kvals[6] = 7.0;
    Kvals[7] = 8.0;
    Kvals[8] = 9.0;
    Kvals[9] = 10.0;
  }
  else if(Ns==24){
    numK = 29;
    if(Kval_index>=numK){
      printf("Error: index out of range.\n");
      exit(-1);
    }
    Kvals = (double*)calloc((size_t)numK, sizeof(double));
    Kvals[0] = 11.0;
    Kvals[1] = 12.0;
    Kvals[2] = 13.0;
    Kvals[3] = 14.0;
    Kvals[4] = 15.0;
    Kvals[5] = 16.0;
    Kvals[6] = 17.0;
    Kvals[7] = 18.0;
    Kvals[8] = 19.0;
    Kvals[9] = 20.0;
    Kvals[10] = 15.2;
    Kvals[11] = 15.4;
    Kvals[12] = 15.6;
    Kvals[13] = 15.8;
    Kvals[14] = 16.2;
    Kvals[15] = 16.4;
    Kvals[16] = 16.6;
    Kvals[17] = 16.8;
    Kvals[18] = 16.7;
    Kvals[19] = 16.9;
    Kvals[20] = 16.81;
    Kvals[21] = 16.82;
    Kvals[22] = 16.83;
    Kvals[23] = 16.84;
    Kvals[24] = 16.85;
    Kvals[25] = 16.86;
    Kvals[26] = 16.87;
    Kvals[27] = 16.88;
    Kvals[28] = 16.89;
  }
  else{
    printf("Size not yet supported.\n");
    exit(-1);
  }

  Kshell = Kvals[Kval_index];

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
  sprintf(input, "/home/frechette/ionex/nc_matrices/stiffshell/hexagon/veff_vec_Ns=%d_nlayer=%d_Kshell=%f.txt", Ns, shell_layers, Kshell);
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
  sprintf(confdir, "confs/nc/stiffshell/kT=%f/h=%f/Ns=%d/c=%f/shell_layers=%d/Kshell=%f/", kT, h, Ns, net_c, shell_layers, Kshell);
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
  sprintf(dir, "pops/nc/stiffshell/kT=%f/h=%f/Ns=%d/c=%f/shell_layers=%d/Kshell=%f/", kT, h, Ns, net_c, shell_layers, Kshell);
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
/*
    if(t<100){
      char outp[300];
      sprintf(outp, "../../phase_sep/nc/stiffshell/confs/equil_%04d.xyz", t);
      pconf(lat, outp, t);
    }
*/
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
      sprintf(outp+strlen(outp), "kmc_conf_%05d_seed=%lu.xyz", t/print_freq, myseed);
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
