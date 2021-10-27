//Perform a Monte Carlo simulation of a 3d elastic nanocrystal

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
int nx = 8;
int ny = 8;
int nz = 8;
int nlayer;
int shell_layers;
double Kshell=1.0;//0.001;
int Ns;

double Lx;
double Ly;
double Lz;

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

  if(argc != 11 and argc!=12){
    printf("Error: require ten arguments (is_arr_job, field, temperature, nsteps, net_c, nx, ny, nz, shell_layers, seed.)\n");
    exit(-1);
  }
  else{
    is_arr_job = atoi(argv[1]);
    h = atof(argv[2]);
    //J = atof(argv[3]);
    kT = atof(argv[3]);
    nsteps = atoi(argv[4]);
    net_c = atof(argv[5]);
    nx = atoi(argv[6]);
		ny = atoi(argv[7]);
		nz = atoi(argv[8]);
    shell_layers = atoi(argv[9]);
    myseed = atoi(argv[10]);
		if(argc==12) Kshell = atof(argv[11]);
  }

	Ns = nx;

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
  char lat_type[100] = "cube";
  int Next_x = nx+shell_layers;
  int Next_y = ny+shell_layers;
  int Next_z = nz+shell_layers;
  if(strcmp(lat_type, "cube")==0){
    Nlat = Next_x*Next_y*Next_z;
    Ncore = nx*ny*nz;
  }
  print_freq = 1;

	//Set up random number generator
	init_rng(myseed);

  //Get value of K_shell from index
  double *Kvals;
  if(Ns==8){
    numK = 34;
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
    Kvals[10] = 0.5;
    Kvals[11] = 0.75;
    Kvals[12] = 1.25;
    Kvals[13] = 1.5;
    Kvals[14] = 1.75;
    Kvals[15] = 2.25;
    Kvals[16] = 2.5;
    Kvals[17] = 2.75;
    Kvals[18] = 1.1;
    Kvals[19] = 1.2;
    Kvals[20] = 1.3;
    Kvals[21] = 1.4;
    Kvals[22] = 1.6;
    Kvals[23] = 1.7;
    Kvals[24] = 1.8;
    Kvals[25] = 1.9;
    Kvals[26] = 2.1;
    Kvals[27] = 2.2;
    Kvals[28] = 2.3;
    Kvals[29] = 2.4;
    Kvals[30] = 2.6;
    Kvals[31] = 2.7;
    Kvals[32] = 2.8;
    Kvals[33] = 2.9;
  }
  else if(Ns==10){
    numK = 34;
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
//    Kvals[10] = 0.001;
    Kvals[10] = 0.5;
    Kvals[11] = 0.75;
    Kvals[12] = 1.25;
    Kvals[13] = 1.5;
    Kvals[14] = 1.75;
    Kvals[15] = 2.25;
    Kvals[16] = 2.5;
    Kvals[17] = 2.75;
    Kvals[18] = 1.1;
    Kvals[19] = 1.2;
    Kvals[20] = 1.3;
    Kvals[21] = 1.4;
    Kvals[22] = 1.6;
    Kvals[23] = 1.7;
    Kvals[24] = 1.8;
    Kvals[25] = 1.9;
    Kvals[26] = 2.1;
    Kvals[27] = 2.2;
    Kvals[28] = 2.3;
    Kvals[29] = 2.4;
    Kvals[30] = 2.6;
    Kvals[31] = 2.7;
    Kvals[32] = 2.8;
    Kvals[33] = 2.9;
  }
  else if(Ns==12){
    numK = 34;
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
    Kvals[10] = 0.5;
    Kvals[11] = 0.75;
    Kvals[12] = 1.25;
    Kvals[13] = 1.5;
    Kvals[14] = 1.75;
    Kvals[15] = 2.25;
    Kvals[16] = 2.5;
    Kvals[17] = 2.75;
    Kvals[18] = 1.1;
    Kvals[19] = 1.2;
    Kvals[20] = 1.3;
    Kvals[21] = 1.4;
    Kvals[22] = 1.6;
    Kvals[23] = 1.7;
    Kvals[24] = 1.8;
    Kvals[25] = 1.9;
    Kvals[26] = 2.1;
    Kvals[27] = 2.2;
    Kvals[28] = 2.3;
    Kvals[29] = 2.4;
    Kvals[30] = 2.6;
    Kvals[31] = 2.7;
    Kvals[32] = 2.8;
    Kvals[33] = 2.9;
  }
  else if(Ns==14 || Ns==16){
    numK = 21;
    if(Kval_index>=numK){
      printf("Error: index out of range.\n");
      exit(-1);
    }
    Kvals = (double*)calloc((size_t)numK, sizeof(double));
    for(int i=0; i<numK; i++) Kvals[i] = 2.0+0.1*i;
  }
  else{
		numK = 1;
		Kvals = (double*)calloc((size_t)numK, sizeof(double));
		Kvals[0] = 10.0;
    //printf("Size not yet supported.\n");
    //exit(-1);
  }

  if(is_arr_job) Kshell = Kvals[Kval_index];
  printf("Kshell: %f\n", Kshell);

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
  char input[2000];
  sprintf(input, "/home/frechette/ionex/nc_matrices/stiffshell/cube/veff_vec_nx=%d_ny=%d_nz=%d_nlayer=%d_Kshell=%f.txt", nx, ny, nz, shell_layers, Kshell);
	std::cout << input << std::endl;
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
  sprintf(confdir, "../confs/3d/stiffshell/kT=%f/h=%f/nx=%d_ny=%d_nz=%d/c=%f/shell_layers=%d/Kshell=%f/", kT, h, nx, ny, nz, net_c, shell_layers, Kshell);
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
  sprintf(dir, "../pops/3d/stiffshell/kT=%f/h=%f/nx=%d_ny=%d_nz=%d/c=%f/shell_layers=%d/Kshell=%f/", kT, h, nx, ny, nz, net_c, shell_layers, Kshell);
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
