//Perform a Kinetic Monte Carlo simulation
//mimicking cation exchange in a 3d simple cubic nanocrystal.
//
//There are two relevant events: irreversibly
//changing the idenitity of a surface atom,
//and exchanging the identities of two adjacent atoms.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_rng.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <armadillo>
#include "kmc.h"
#include "elastic.h"
#include "atom.h"
#include "shared.h"

#define Z 18 //Change for different lattice structures

//Global Variables
gsl_rng *rg; //Random number generation

//Set basic timescales of "exchange" and "diffusion"
double k_flip = 1.0; //Rate of changing identity of surface atom
double k_swap = 1.0; //Rate of exchanging identity of adjacent atoms
int k_swap_pow = 0;

//Elastic variables
double laa = 1.0;
double lbb = 0.8;
double K = 200;

int nsteps = 1000;
int nsweep;
int print_freq;

int Nlat;
int nx, ny, nz;
int Nup;
int Nsurf;
int Npair=0;

double h = 5.0;
double J = 0.0;
double kT = 2.0;

long unsigned int myseed = 1;
int is_no_ex = 0;

const char *init;
std::string init_short;

int is_arr_job;

static void _mkdir(const char *dir);

/*** Main ***/

int main(int argc, char *argv[]){

  if(argc != 12){
    printf("Error: require nine arguments (is_arr_job, nx, ny, nz, kT, h, k_swap_pow, J, nsweep, init, is_no_ex.)\n");
    exit(-1);
  }
  else{
    is_arr_job = atoi(argv[1]);
    nx = atoi(argv[2]);
    ny = atoi(argv[3]);
    nz = atoi(argv[4]);
    kT = atof(argv[5]);
    h = atof(argv[6]);
    k_swap_pow = atoi(argv[7]);
    J = atof(argv[8]);
    nsweep = atoi(argv[9]);
    init = argv[10];
    is_no_ex = atoi(argv[11]);
  }

	std::cout << init << std::endl;

	std::string init_plus(init);
	std::size_t myindex = init_plus.find_last_of("/");
	init_short = init_plus.substr(myindex+1);
	myindex = init_short.find(".");
	init_short = init_short.substr(0,myindex);
	std::cout << init_short << std::endl;

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
  
    myseed = taskID;
  }
  else{
    myseed = 1;
  }

  //Compute global variables
  Nlat = nx*ny*nz; 
  Nup = Nlat; //Start out with all spins up
  Nsurf = nx*ny*nz-(nx-2)*(ny-2)*(nz-2); //is this true for non-cube?

	k_swap = pow(10.0, k_swap_pow);
  nsteps = Nlat*nsweep;
  print_freq = Nlat;//nsteps/(Nlat*10);

  if(is_no_ex) k_flip=0.0;

  //Set up random number generator
  init_rng(myseed);

  //Memory allocation
  std::vector<atom> lat(Nlat);
  arma::mat veff(Nlat, Nlat, arma::fill::zeros);
  double *fraction = (double*)calloc((size_t)nsteps, sizeof(double));
  double *time = (double*)calloc((size_t)nsteps, sizeof(double));
  int *surf_ind = (int*)calloc((size_t)Nsurf, sizeof(int));
	
  //Set up nanocrystal
  initialize(lat, surf_ind, init);
  std::vector< std::pair <int, int> > pair_ind(Npair);
  if(get_pair_indices(lat, pair_ind)!=Npair){
    printf("Error: pair indexing issue.\n");
    exit(0);
  }

  //Get effective potential
  printf("Reading effective potential...\n");
  char input[1000];
  sprintf(input, "veff_vec_nx=%d_ny=%d_nz=%d.txt", nx, ny, nz);
  struct stat fileStat0;
  if(stat(input, &fileStat0)!=0){
    printf("Error: effective potential file does not exist.\n");
    exit(-1);
  }
  read_potential(input, veff);
  printf("Read in effective potential.\n");
	
  //Run KMC
  printf("Running trajectory with seed %lu.\n", myseed);
  run_traj_elastic(lat, fraction, time, surf_ind, pair_ind, nsteps, veff);

  //Print Data
  print_pop(time, fraction, nsteps);

  return 1;
}

/***************************/
/*** Key Functions ***/
/***************************/

double run_traj_elastic(std::vector<atom> &lat, double *fraction, double *time,
                int *surf_ind, std::vector< std::pair <int, int> > &pair_ind, int nstp,
                arma::mat &veff){

  /*** First, create array of all possible events and their rates. 
   *** The rates will be updated after each event. ***/

  //Possible events:
  //1) Surface spin 1 -> -1
  //2) Surface spin -1 -> 1
  //3) Swap any two adjacent spins

	/***Generate array of partial sums***/
  int event_curr = 0;
  int nevents = Nsurf + Npair;

  double *partsum = (double*)calloc((size_t)nevents, sizeof(double)); //Store partial sums
  int *event_ind = (int*)calloc((size_t)nevents, sizeof(int)); //Index from event list to surface site/pair

  //Loop through surface sites and find potential spin flips
  for(int i=0; i<Nsurf; i++){
    event_ind[event_curr] = i; //update event list
    partsum[event_curr] = k_flip*get_boltzmann_flip_elastic(lat, surf_ind[i], veff);
    if(event_curr>0) partsum[event_curr] += partsum[event_curr-1];
    event_curr++;
  }
  //Loop through pairs of sites and find potential swap moves
  for(int i=0; i<Npair; i++){
    event_ind[event_curr] = i;
    partsum[event_curr] = k_swap*get_boltzmann_swap_elastic(lat, pair_ind[i].first,
                                                           pair_ind[i].second, veff);
    if(event_curr>0) partsum[event_curr] += partsum[event_curr-1];
    event_curr++;
  }

  double ktot = partsum[nevents-1];

  double t_curr = 0;

	for(int t=0; t<nstp; t++){

    //Print configurations for first Nlat moves,
    //then only print every print_freq moves
    if(t<Nlat){
      std::cout << "Pass number " << t << "\n";
      //Check if directory exists
      char dir[200];
		  sprintf(dir, "data/init=%s/J=%f/nx=%d_ny=%d_nz=%d/kT=%f/h=%f/k_swap=1e%d/confs/seed=%03d/", init_short.c_str(), J, nx, ny, nz, kT, h, k_swap_pow, myseed);
		  struct stat fileStat;
		  if(stat(dir, &fileStat)<0){
			  _mkdir(dir);
			  printf("Created directory.\n");
		  }

		  char outp[200];
		  sprintf(outp, "data/init=%s/J=%f/nx=%d_ny=%d_nz=%d/kT=%f/h=%f/k_swap=1e%d/confs/seed=%03d/shortt_%04d.xyz", init_short.c_str(), J, nx, ny, nz, kT, h, k_swap_pow, myseed, t);
		  //pconf(lat, outp, t_curr);
    }

    //Print configurations
    if(t%print_freq == 0){
      std::cout << "Pass number " << t/Nlat << "\n";
      //Check if directory exists
      char dir[200];
        sprintf(dir, "data/init=%s/J=%f/nx=%d_ny=%d_nz=%d/kT=%f/h=%f/k_swap=1e%d/confs/seed=%03d/", init_short.c_str(), J, nx, ny, nz, kT, h, k_swap_pow, myseed);
        struct stat fileStat;
        if(stat(dir, &fileStat)<0){
          //mkdir(dir, ACCESSPERMS);
          _mkdir(dir);
          printf("Created directory.\n");
        }

        char outp[100];
        sprintf(outp, "data/init=%s/J=%f/nx=%d_ny=%d_nz=%d/kT=%f/h=%f/k_swap=1e%d/confs/seed=%03d/conf_%04d.xyz", init_short.c_str(), J, nx, ny, nz, kT, h, k_swap_pow, myseed, t/print_freq);
        pconf(lat, outp, t_curr);
    } 
    
		  //Select event from list
      event_curr = 0;
      double u1 = gsl_rng_uniform(rg);
      double pdt = u1*ktot;
      //linear search -- could do better
      for(int i=0; i<nevents; i++){
        if(partsum[i]>pdt){
          event_curr = i;
          break;
        }
      }

      //Check whether event is swap between two identical spins.
      //If so, advance time counter but don't count as one of the
      //nstps moves.
      
      if(event_curr>=Nsurf && lat[pair_ind[event_ind[event_curr]].first].s==lat[pair_ind[event_ind[event_curr]].second].s){
        t_curr += log(1.0/gsl_rng_uniform(rg))/ktot;
        time[t] = t_curr;
        fraction[t] = (1.0*get_nup(lat))/(1.0*Nlat);
        continue;
      }
//      else{
//        found_good_event=1;
     // }
      
      //found_good_event=1;
//    } 

    //Execute event
    if(event_curr<Nsurf){
      lat[surf_ind[event_ind[event_curr]]].s *= -1; 
    }
    else{
      int temp = lat[pair_ind[event_ind[event_curr]].first].s;
      lat[pair_ind[event_ind[event_curr]].first].s = lat[pair_ind[event_ind[event_curr]].second].s; 
      lat[pair_ind[event_ind[event_curr]].second].s = temp; 
    }
	
		//Advance time by drawing random number
    //from exponential distribution.
    double u2 = gsl_rng_uniform(rg);
    t_curr += log(1.0/u2)/ktot;

    //Record data
    time[t] = t_curr;
    fraction[t] = (1.0*get_nup(lat))/(1.0*Nlat); 

    //Update rates
    event_curr=0;
    for(int i=0; i<Nsurf; i++){
      partsum[event_curr] = k_flip*get_boltzmann_flip_elastic(lat, surf_ind[i], veff);
      if(event_curr>0) partsum[event_curr] += partsum[event_curr-1];
      event_curr++;
    }
    for(int i=0; i<Npair; i++){
      partsum[event_curr] = k_swap*get_boltzmann_swap_elastic(lat, pair_ind[i].first,
                                                           pair_ind[i].second, veff);
      partsum[event_curr] += partsum[event_curr-1];
      event_curr++;
    }
    ktot = partsum[nevents-1];

	}

  //Free memory
  free(partsum);
  partsum =NULL;

  return 1;
}

/*********************************************/

void initialize(std::vector<atom> &lat, int *surf_ind, const char *init_conf){

  //Set spin values
  if(strcmp(init_conf,"ones")==0){
    std::cout << "Initializing with all spins up." << std::endl;
  	for(int i=0; i<Nlat; i++) lat[i].s = 1;

	  //Set position values
  	int cnt = 0;
	  double lavg = 1.0;

  	double a_lat = lavg;

	  for(int i=0; i<nx; i++){
  	  for(int j=0; j<ny; j++){
    	  for(int k=0; k<nz; k++){

      	  lat[cnt].pos(0) = a_lat*i;
        	lat[cnt].pos(1) = a_lat*j;
	        lat[cnt].pos(2) = a_lat*k;

  	      cnt+=1;
	      }
  	  }
	  }
  }
	else{
		std::cout << "Attempting to read spins from file." << std::endl;
    FILE *f;
    f = fopen(init_conf, "r");
		printf("opened file\n");
		int trashnum;
		char trash[100];
		double trashtime;
    char atom[2];
		if(fscanf(f, "%d\n", &trashnum) != 1) printf("Error: need 1 number.\n");
		std::cout << trashnum << std::endl;
		if(fscanf(f, "%s %lf\n", trash, &trashtime) != 2) printf("Error: need string.\n");
		std::cout << trash << std::endl;
		std::cout << trashtime << std::endl;
		for(int i=0; i<Nlat; i++){
			if(fscanf(f, "%s %lf %lf %lf\n", atom, &lat[i].pos(0), &lat[i].pos(1), &lat[i].pos(2)) != 4) printf("Error: need 4 inputs.\n");
			if(strcmp(atom,"Ar")==0) lat[i].s = 1;
			else if(strcmp(atom,"S")==0) lat[i].s = -1;
			else std::cout << "Atom not recognized!!!" << std::endl;
		}
  	fclose(f);
  	f=NULL;
	}

	//Get nearest neighbors
	for(int i=0; i<Nlat; i++){
		int nn=0;
		for(int j=0; j<Nlat; j++){
			if(i!=j){
				if(get_dist(lat[i], lat[j])<(sqrt(2.0)*laa+1e-4)){
          if(j>i) Npair++;
					lat[i].nbs.push_back(j);
					nn++;
				}
			}
		}
    lat[i].n = nn;
	}

  //Index surface atoms
  int indno = 0;
  for(int i=0; i<Nlat; i++){
    if(lat[i].n<Z){
      //Check to make sure not going out-of-bounds
      if(indno>Nsurf){
        printf("Error: trying to index non-existent site.\n");
        exit(0);
      }
      surf_ind[indno] = i;
      indno++;
    }
  }

}

/*********************************************/

int get_pair_indices(std::vector<atom> &lat, std::vector< std::pair <int, int> > &pair_ind){

  int ind_curr = 0;

	for(int i=0; i<Nlat; i++){
		for(int j=0; j<lat[i].n; j++){
			if(lat[i].nbs[j]>i){
        pair_ind[ind_curr].first = i;
        pair_ind[ind_curr].second = lat[i].nbs[j];
        ind_curr++;
			}
		}
	}
  return ind_curr; 
}

/**************************/
/*** Energy Functions ***/
/**************************/

double get_nup(std::vector<atom> &lat){

  double nup = 0;
  for(int i=0; i<Nlat; i++){
    if(lat[i].s==1) nup++;
  }
  return nup;
}

double get_field_energy(std::vector<atom> &lat){

  double mag = 0;
  for(int i=0; i<Nlat; i++) mag += lat[i].s;
  return -h*mag;
}

double get_pair_energy(std::vector<atom> &lat){

  double ene = 0;
  for(int i=0; i<Nlat; i++){
    for(int j=0; j<lat[i].n; j++){
      if(j>i){
        ene += lat[i].s*lat[lat[i].nbs[j]].s;
      }
    }
  }
  return (-J)*ene;
}

double get_elastic_energy(std::vector<atom> &lat, arma::mat &veff){

  arma::vec spinvec(Nlat, arma::fill::zeros);
  arma::vec myones(Nlat, arma::fill::ones);
  for (int i=0; i<Nlat; i++){
    spinvec(i) = lat[i].s;
  }
  spinvec = spinvec - myones*mean(spinvec);

  return dot(spinvec, veff*spinvec);
}

double get_elastic_flip(std::vector<atom> &lat, arma::mat &veff, int j){

  arma::vec spinvec(Nlat, arma::fill::zeros);
  arma::vec myones(Nlat, arma::fill::ones);
  for (int i=0; i<Nlat; i++){
    spinvec(i) = lat[i].s;
  }
  double m = mean(spinvec);
  spinvec = spinvec - myones*m;

  return (lat[j].s-m)*dot(veff.row(j), spinvec);
}

double get_elastic_swap(std::vector<atom> &lat, arma::mat &veff, int j, int k){

  arma::vec spinvec(Nlat, arma::fill::zeros);
  arma::vec myones(Nlat, arma::fill::ones);
  for (int i=0; i<Nlat; i++){
    spinvec(i) = lat[i].s;
  }
  double m = mean(spinvec);
  spinvec = spinvec - myones*m;

  return (lat[j].s-m)*dot(veff.row(j), spinvec)
       + (lat[k].s-m)*dot(veff.row(k), spinvec)
       - (lat[j].s-m)*veff(j,k)*(lat[k].s-m);
}

double get_boltzmann_flip_elastic(std::vector<atom> &lat, int ind, arma::mat &veff){
  
  double E1 = -h*lat[ind].s;
  for(int j=0; j<lat[ind].n; j++) E1 += (-J)*lat[ind].s*lat[lat[ind].nbs[j]].s;
  E1 += get_elastic_flip(lat, veff, ind);
  lat[ind].s = -lat[ind].s;
  double E2 = -h*lat[ind].s;
  for(int j=0; j<lat[ind].n; j++) E2 += (-J)*lat[ind].s*lat[lat[ind].nbs[j]].s;
  E2 += get_elastic_flip(lat, veff, ind);
  lat[ind].s = -lat[ind].s;

  return exp(-(E2-E1)/(2.0*kT));

}

double get_boltzmann_swap_elastic(std::vector<atom> &lat, int ind1, int ind2, arma::mat &veff){

  double E1 = 0;
  for(int j=0; j<lat[ind1].n; j++){
    if(lat[ind1].nbs[j] != ind2) E1 += (-J)*lat[ind1].s*lat[lat[ind1].nbs[j]].s;
  }
  for(int j=0; j<lat[ind2].n; j++){
    if(lat[ind2].nbs[j] != ind1) E1 += (-J)*lat[ind2].s*lat[lat[ind2].nbs[j]].s;
  } 
  E1 += get_elastic_swap(lat, veff, ind1, ind2);
  int temp = lat[ind1].s;
  lat[ind1].s = lat[ind2].s;
  lat[ind2].s = temp;
  double E2 = 0;
  for(int j=0; j<lat[ind1].n; j++){
    if(lat[ind1].nbs[j] != ind2) E2 += (-J)*lat[ind1].s*lat[lat[ind1].nbs[j]].s;
  }
  for(int j=0; j<lat[ind2].n; j++){
    if(lat[ind2].nbs[j] != ind1) E2 += (-J)*lat[ind2].s*lat[lat[ind2].nbs[j]].s;
  } 
  E2 += get_elastic_swap(lat, veff, ind1, ind2);
  int temp2 = lat[ind1].s;
  lat[ind1].s = lat[ind2].s;
  lat[ind2].s = temp2;

  return exp(-(E2-E1)/(2.0*kT));
}

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
    }
  }
  fclose(f);
  f=NULL;
}

/**************************/
/*** Printing Functions ***/
/**************************/

void print_pop(double *time, double *fraction, int nstp){

  //Check if directory exists
  char dir[200];
  sprintf(dir, "data/init=%s/J=%f/nx=%d_ny=%d_nz=%d/kT=%f/h=%f/k_swap=1e%d/pops/", init_short.c_str(), J, nx, ny, nz, kT, h, k_swap_pow);
  struct stat fileStat;
  if(stat(dir, &fileStat)<0){
    _mkdir(dir);
    printf("Created directory.\n");
  }

  //Write file
	FILE *f;
  char outp[200];
  sprintf(outp, "data/init=%s/J=%f/nx=%d_ny=%d_nz=%d/kT=%f/h=%f/k_swap=1e%d/pops/kmc_pop_seed=%lu.txt", init_short.c_str(), J, nx, ny, nz, kT, h, k_swap_pow, myseed);
  f = fopen(outp, "w");
	for(int i=0; i<nstp; i++){
		fprintf(f, "%.15e %f\n", time[i], fraction[i]);
	}
	fclose(f);
	f=NULL;
}

void pconf(std::vector<atom> &lat, const char *name, double time){

  FILE *f;
	f = fopen(name, "w");
	fprintf(f, "%d\n", Nlat);
	fprintf(f, "%s %.15e\n", "time", time);

	for(int i=0; i<Nlat; i++){
		if(lat[i].s==1) fprintf(f, "Ar %f %f %f\n", lat[i].pos(0), lat[i].pos(1), lat[i].pos(2)); 
    else fprintf(f, "S %f %f %f\n", lat[i].pos(0), lat[i].pos(1), lat[i].pos(2));
	}
	fclose(f);
	f=NULL;
}

/*******************************/
/*** Miscellaneous Functions ***/
/*******************************/

void init_rng(long unsigned int seed){
  
	const gsl_rng_type *T;
	T = gsl_rng_default;
	rg = gsl_rng_alloc(T);
	srand((unsigned) seed);
	gsl_rng_env_setup();
	gsl_rng_set(rg, seed);
}

//This function was copied from stackoverflow
//users Carl Norum and Basic
static void _mkdir(const char *dir) {
        char tmp[256];
        char *p = NULL;
        size_t len;

        snprintf(tmp, sizeof(tmp),"%s",dir);
        len = strlen(tmp);
        if(tmp[len - 1] == '/')
                tmp[len - 1] = 0;
        for(p = tmp + 1; *p; p++)
                if(*p == '/') {
                        *p = 0;
                        mkdir(tmp, S_IRWXU);
                        *p = '/';
                }
        mkdir(tmp, S_IRWXU);
}
