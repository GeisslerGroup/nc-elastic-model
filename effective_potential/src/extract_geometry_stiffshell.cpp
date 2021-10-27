/* Extract effective potential of 2d triangular lattice NC */

//C++ headers
#include <iostream>
#include <armadillo>
#include <iomanip> 

//C headers
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_rng.h>

//User-defined headers
#include "elastic_ising.h"
#include "site.h"

//Set global constants
#define J  0.0 //coupling constant
#define npass 1e5
#define nfreq 100

gsl_rng *rg; //use gsl for random number generation

int Nlat;
int num_surf;
int Ns = 3;  //length of side of hexagon
int nx=4;
int ny=20;
double dx = 0.1;

double H = 0.0;
double kT = 0.1;

double laa = 1.0;
double lbb = 0.8;
double lab;

double Kshell = 10.0;
double K = 200.0;
double kaa = K;
double kbb = K;
double kab;

int disp_acc = 0;
int spin_acc = 0;

int fixed = 0;
int fix_boundary = 0;
int boundary_length_type = 0;
int boundary_type_b = 0;

int nlayer = 0;

int center_index = -1;

char shape[] = "hexagon";

void initialize_atoms(std::vector<site> &lat, char *init);
void initialize_bond(std::vector<site> &lat, int **diagonal, int **off_diag, arma::mat &rotkin);
void boys_min(std::vector<site> &lat, arma::mat eigvec, arma::vec coeffs);
int is_neighbor(std::vector<site> &lat, int m1, int m2);
void pbonddata(int **d, int **od);
void pmatrices(arma::mat dyn, arma::mat coup, arma::mat spin, arma::mat veff);
void pconfigs(std::vector<site> &lat, arma::mat veff);
void pevals(arma::vec evals);
void pevecs(arma::mat evecs, std::vector<site> &lat);
void pfonevals(arma::vec evals);

/******************************************************/
/*** MAIN ***/
/******************************************************/

int main(int argc, char *argv[]){

  if(argc!=4){
    printf("Error: need size of hexagon, shell stiffness, no. of shell layers.\n");
    exit(-1);
  }
  
  Ns = atoi(argv[1]);
  Kshell = atof(argv[2]);
  nlayer = atoi(argv[3]);

	//Compute global varaibles
  int Next = Ns+nlayer;
	Nlat = 3*Next*Next - 3*Next + 1;
  //int Ncore = 3*Ns*Ns-3*Ns+1;
  if(strcmp(shape, "long")==0) Nlat = nx*ny;
  kab = 0.5*(kaa+kbb);
  lab = 0.5*(laa+lbb);

  char init[] = "ones";

  //Initialize random number generator
  const gsl_rng_type * t;

  t = gsl_rng_default;
  rg = gsl_rng_alloc (t);

  //Set random seed
  long unsigned int seed = 4;
  srand((unsigned) seed);
  gsl_rng_env_setup();
  gsl_rng_set(rg, seed);

  //Define triangular lattice vectors
  arma::vec a1 = {1.0/2.0, sqrt(3.0)/2.0};
  arma::vec a2 = {1.0, 0};
  arma::vec a3 = {1.0/2.0, -sqrt(3.0)/2.0};

  arma::mat alpha_mat(3,2);
  alpha_mat(0,0) = a1(0);
  alpha_mat(0,1) = a1(1);
  alpha_mat(1,0) = a2(0);
  alpha_mat(1,1) = a2(1);
  alpha_mat(2,0) = a3(0);
  alpha_mat(2,1) = a3(1);

  //Allocate memory
	std::vector<site> lat(Nlat);
  initialize_atoms(lat, init);
  int **diagonal = (int**)calloc((size_t)Nlat, sizeof(int*));
  for(int i=0; i<Nlat; i++){
    diagonal[i] = (int*)calloc((size_t)7, sizeof(int));
  }
  int **off_diag = (int**)calloc((size_t)Nlat, sizeof(int*));
  for(int i=0; i<Nlat; i++){
    off_diag[i] = (int*)calloc((size_t)Nlat, sizeof(int));
  }
  arma::mat full(3*Nlat, 3*Nlat, arma::fill::zeros);
  arma::mat dyn(2*Nlat, 2*Nlat, arma::fill::zeros);
  arma::mat coup(2*Nlat, Nlat, arma::fill::zeros);
  arma::mat spin(Nlat, Nlat, arma::fill::zeros); 
  arma::mat spin_mod(Nlat, Nlat, arma::fill::zeros);
  arma::mat veff(Nlat, Nlat, arma::fill::zeros);
  arma::mat veff_mod(Nlat, Nlat, arma::fill::zeros);
  arma::mat ising(Nlat, Nlat, arma::fill::zeros);
  arma::mat rotkin(Nlat, Nlat, arma::fill::zeros);
  arma::vec spinvec(Nlat, arma::fill::zeros);
  arma::vec fullvec(3*Nlat, arma::fill::zeros);
  arma::vec myones(Nlat, arma::fill::zeros);
  for(int i=0; i<Nlat; i++){
    spinvec(i) = 1.0;
    myones(i) = 1.0;
  }

  //Create lattice 
  initialize_bond(lat, diagonal, off_diag, rotkin);
  printf("bonds initialized\n");

  /* Fill in matrices */
  for(int i=0; i<Nlat; i++){
    //Spin Matrix
    for(int k=0; k<lat[i].n; k++){
      double Ktemp = 1.0;
      if(lat[(int)lat[i].nbs(k)].layer<=nlayer && lat[i].layer<=nlayer) Ktemp = Kshell;
      spin(i, i) += Ktemp/4.0;
    }
    for(int j=0; j<Nlat; j++){
      double Ktemp = 1.0;
      if(lat[j].layer<=nlayer && lat[i].layer<=nlayer) Ktemp = Kshell;
      if(i!=j){
        if(off_diag[i][j]!=0){
          spin(i,j)=Ktemp/4.0;
          spin_mod(i,j)=Ktemp/4.0;
        }
      }
    }
    //Dynamical Matrix
    for(int k=0; k<lat[i].n; k++){
      double Ktemp = 1.0;
      if(lat[(int)lat[i].nbs(k)].layer<=nlayer && lat[i].layer<=nlayer) Ktemp = Kshell;
      dyn(2*i,2*i) += Ktemp*alpha_mat(abs(diagonal[i][k+1])-1, 0)*alpha_mat(abs(diagonal[i][k+1])-1, 0);
      dyn(2*i,2*i+1) += Ktemp*alpha_mat(abs(diagonal[i][k+1])-1, 0)*alpha_mat(abs(diagonal[i][k+1])-1, 1);
      dyn(2*i+1,2*i) += Ktemp*alpha_mat(abs(diagonal[i][k+1])-1, 1)*alpha_mat(abs(diagonal[i][k+1])-1, 0);
      dyn(2*i+1,2*i+1) += Ktemp*alpha_mat(abs(diagonal[i][k+1])-1, 1)*alpha_mat(abs(diagonal[i][k+1])-1, 1);
    }
    for(int j=0; j<Nlat; j++){
      double Ktemp = 1.0;
      if(lat[j].layer<=nlayer && lat[i].layer<=nlayer) Ktemp = Kshell;
      if(i!=j && is_neighbor(lat, i, j)){
        dyn(2*i, 2*j) = -Ktemp*alpha_mat(abs(off_diag[i][j])-1, 0)*alpha_mat(abs(off_diag[i][j])-1, 0);
        dyn(2*i, 2*j+1) = -Ktemp*alpha_mat(abs(off_diag[i][j])-1, 0)*alpha_mat(abs(off_diag[i][j])-1, 1);
        dyn(2*i+1, 2*j) = -Ktemp*alpha_mat(abs(off_diag[i][j])-1, 1)*alpha_mat(abs(off_diag[i][j])-1, 0);
        dyn(2*i+1, 2*j+1) = -Ktemp*alpha_mat(abs(off_diag[i][j])-1, 1)*alpha_mat(abs(off_diag[i][j])-1, 1);
      }
    }
    //Coupling Matrix
    for(int k=0; k<lat[i].n; k++){
      double Ktemp = 1.0;
      if(lat[(int)lat[i].nbs(k)].layer<=nlayer && lat[i].layer<=nlayer) Ktemp = Kshell;
      coup(2*i, i) += Ktemp*alpha_mat(abs(diagonal[i][k+1])-1, 0)*copysign(1.0, -diagonal[i][k+1]);
      coup(2*i+1, i) += Ktemp*alpha_mat(abs(diagonal[i][k+1])-1, 1)*copysign(1.0, -diagonal[i][k+1]);
    }
    for(int j=0; j<Nlat; j++){
      double Ktemp = 1.0;
      if(lat[j].layer<=nlayer && lat[i].layer<=nlayer) Ktemp = Kshell;
      if(i!=j) ising(i,j) = -4.0/Nlat;
      if(i!=j && is_neighbor(lat, i, j)){
        coup(2*i, j) += Ktemp*alpha_mat(abs(off_diag[i][j])-1, 0)*copysign(1.0, -off_diag[i][j]);
        coup(2*i+1, j) += Ktemp*alpha_mat(abs(off_diag[i][j])-1, 1)*copysign(1.0, -off_diag[i][j]);
        //ising(i,j) += 0.01;
      }
    }
    //Full Matrix
    for(int k=0; k<lat[i].n; k++){
      full(3*i,3*i) += alpha_mat(abs(diagonal[i][k+1])-1, 0)*alpha_mat(abs(diagonal[i][k+1])-1, 0);
      full(3*i,3*i+1) += alpha_mat(abs(diagonal[i][k+1])-1, 0)*alpha_mat(abs(diagonal[i][k+1])-1, 1);
      full(3*i+1,3*i) += alpha_mat(abs(diagonal[i][k+1])-1, 1)*alpha_mat(abs(diagonal[i][k+1])-1, 0);
      full(3*i+1,3*i+1) += alpha_mat(abs(diagonal[i][k+1])-1, 1)*alpha_mat(abs(diagonal[i][k+1])-1, 1);
    }
    for(int j=0; j<Nlat; j++){
      if(i!=j && is_neighbor(lat, i, j)){
        full(3*i, 3*j) = -alpha_mat(abs(off_diag[i][j])-1, 0)*alpha_mat(abs(off_diag[i][j])-1, 0);
        full(3*i, 3*j+1) = -alpha_mat(abs(off_diag[i][j])-1, 0)*alpha_mat(abs(off_diag[i][j])-1, 1);
        full(3*i+1, 3*j) = -alpha_mat(abs(off_diag[i][j])-1, 1)*alpha_mat(abs(off_diag[i][j])-1, 0);
        full(3*i+1, 3*j+1) = -alpha_mat(abs(off_diag[i][j])-1, 1)*alpha_mat(abs(off_diag[i][j])-1, 1);
      }
    }
    for(int k=0; k<lat[i].n; k++){
      full(3*i, 3*i+2) += alpha_mat(abs(diagonal[i][k+1])-1, 0)*copysign(1.0, -diagonal[i][k+1]);
      full(3*i+1, 3*i+2) += alpha_mat(abs(diagonal[i][k+1])-1, 1)*copysign(1.0, -diagonal[i][k+1]);
    }
    for(int j=0; j<Nlat; j++){
      if(i!=j && is_neighbor(lat, i, j)){
        full(3*i, 3*j+2) += alpha_mat(abs(off_diag[i][j])-1, 0)*copysign(1.0, -off_diag[i][j]);
        full(3*i+1, 3*j+2) += alpha_mat(abs(off_diag[i][j])-1, 1)*copysign(1.0, -off_diag[i][j]);
      }
    }
    full(3*i+2, 3*i+2) = diagonal[i][0]/4.0;
    for(int j=0; j<Nlat; j++){
      if(i!=j){
        if(off_diag[i][j]!=0) full(3*i+2,3*j+2)=1/4.0;
      }
    }
  }

  /* Construct V_eff. Need to remove zero modes. */

  dyn.shed_row(2*(Ns+nlayer)-1);
  dyn.shed_col(2*(Ns+nlayer)-1);
  dyn.shed_rows(0,1);
  dyn.shed_cols(0,1);
  coup.shed_row(2*(Ns+nlayer)-1);
  coup.shed_rows(0,1);

  veff = 4.0*spin - coup.t()*inv(dyn)*coup;
  veff_mod = 4.0*spin_mod - coup.t()*inv(dyn)*coup;
  
  //double temp = veff(0,0);
  /*
  for(int i=0; i<Nlat; i++){
    for(int j=0; j<Nlat; j++){
      veff(i,j)=veff(i,j)-12.0/Nlat;
    }
  }
  */

  printf("Obtained effective potential\n");
  printf("Sum of Veff elements: %.15e\n", accu(veff));

  //Obtain Q matrix
  //First diagonalize full matrix
  arma::vec full_eval;
  arma::mat full_evec = arma::zeros(3*Nlat, 3*Nlat);
  eig_sym(full_eval, full_evec, full);
  printf("Diagonalized full matrix.\n");
  //std::cout << full_eval << std::endl;
  //std::cout << full_evec << std::endl;

  //Test a configuration
  for(int i=0; i<Nlat; i++){
    fullvec(3*i) = 0;
    fullvec(3*i+1) = 0;
    fullvec(3*i+2) = 1.0;
  }
  //std::cout << fullvec << std::endl;
  //std::cout << "Full vector:\n" << full*fullvec << std::endl;

  for(int i=0; i<Nlat; i++) spinvec(i) = 0;
  spinvec(0) = 1;
  spinvec(3) = 0.15;

  double energy = dot((spinvec-myones*mean(spinvec)), veff*(spinvec-myones*mean(spinvec)));
  printf("energy: %f\n", energy);
  double energy2 = dot((spinvec), veff*(spinvec));
  printf("energy2: %f\n", energy2);

  arma::vec eigval;
  arma::mat eigvec = arma::zeros(Nlat, Nlat);
  arma::vec eigval_mod;
  arma::mat eigvec_mod = arma::zeros(Nlat, Nlat);

  eig_sym(eigval, eigvec, veff);//+3.14159*rotkin);
  eig_sym(eigval_mod, eigvec_mod, veff_mod);
  //for(int i=0; i<Nlat; i++) printf("eval %d: %f\n", i, eigval(i));
  //for(int i=0; i<Nlat; i++) printf("mod eval %d: %f\n", i, eigval_mod(i));
  for(int i=0; i<Nlat; i++) spinvec(i) = -1.0/sqrt(Nlat);
  //std::cout << eigvec.col(0) << std::endl;
  //for(int i=0; i<Nlat; i++) printf("orthogonal? %f\n", dot(eigvec.col(1), eigvec.col(i)));
  //double theta = arccos(dot(eigvec.col(17), spinvec)); 
  //std::cout << veff << std::endl;
  //std::cout << eigvec << std::endl;
  //std::cout << veff*eigvec << std::endl;
 // std::cout << inv(dyn) << std::endl;
 // std::cout << coup << std::endl;
 
  num_surf=0;
  int num_bond = 0;
  for(int i=0; i<Nlat; i++){
    if(lat[i].n<6) num_surf++;
    num_bond += lat[i].n;
  }
  num_bond = num_bond/2;
  printf("num surf: %d\n", num_surf);
  printf("Nlat: %d\n", Nlat);
  printf("num bonds: %d\n", num_bond);
  printf("rule count: %d\n", (2+1)*(Nlat-1)-num_bond);

  //Ising Matrix
  //arma::vec isingval;
  //arma::mat isingvec = arma::zeros(Nlat, Nlat);
  //eig_sym(isingval, isingvec, ising);
  //std::cout << isingvec << std::endl;
  //for(int i=0; i<Nlat; i++) printf("ising eval %d: %f\n", i, isingval(i));

  //Phonon Matrix
  arma::vec phonon_eval;
  arma::mat phonon_evec = arma::zeros(2*Nlat, 2*Nlat);
  //printf("phonon matrix\n");
  eig_sym(phonon_eval, phonon_evec, dyn);
  //std::cout << phonon_eval << std::endl;

  //std::cout << "product" << std::endl;
  //std::cout << std::fixed << std::setprecision(2);
  //std::cout << eigvec_mod << std::endl;
  //std::cout << veff*eigvec_mod << std::endl;
 // std::cout << phonon_eval << std::endl;
  arma::mat myprod = veff*eigvec;

  arma::vec coeffs = arma::zeros(Nlat);
  for(int i=0; i<Nlat; i++) coeffs(i) = 1.0/Nlat;
  //boys_min(lat, eigvec, coeffs);
  
  //std::cout << "%f\n" << sum(veff) << std::endl;
  //std::cout << "%f\n" << accu(veff) << std::endl;

  pmatrices(dyn, coup, spin, veff);
	//pbonddata(diagonal, off_diag);
  //pconfigs(lat, veff);
  //pevals(eigval);
  //pevecs(eigvec, lat);
  //pfonevals(phonon_eval);
}

/* Function Definitions *********************************************************************************/

void initialize_atoms(std::vector<site> &lat, char *init){

	//First Set spin values
  if(strcmp(init, "ones")==0){
    for(int i = 0; i < Nlat; i++) lat[i].s = 1;
  }
  else{
    printf("Not a valid initial configuration.\n");
    exit(-1);
  }

	//Next set position values
	int cnt = 0;
	double xs = 0;
	double ys = 0;
  
  double lavg=laa;

  if(strcmp(shape, "hexagon")==0){

    for(int i=0; i<Ns+nlayer; i++){
      xs = (-lavg/2.0)*i;
      ys = lavg*(sqrt(3.0)/2.0)*i;
      for(int j=0; j<(Ns+nlayer+i); j++){
        //if(j==(Ns+i)-1 || j==0 || i==0){
          //double xsi = gsl_rng_uniform(rg);
          //if(xsi>-1.0){
            lat[cnt].layer = std::min(i+1, std::min(j+1, Ns+nlayer+i-j));
            lat[cnt].x = xs + j*lavg;
            lat[cnt].y = ys;	
            cnt++;
          //}
        //}
        /*
        else{
          lat[cnt].x = xs + j*lavg;
          lat[cnt].y = ys;	
          cnt++;
        }
        */
      }
    }
    for(int i=(Ns+nlayer-2); i>=0; i--){
      xs += lavg/2.0;
      ys += lavg*(sqrt(3.0)/2.0);
      for(int j=0; j<(Ns+nlayer+i); j++){
        //if(j==(Ns+i)-1 || j==0 || i==0){
          //double xsi = gsl_rng_uniform(rg);
          //if(xsi>-1.0){
            lat[cnt].layer = std::min(i+1, std::min(j+1, Ns+nlayer+i-j));
            lat[cnt].x = xs + j*lavg;
            lat[cnt].y = ys;	
            cnt++;
          //}
        //}
        /*
        else{
          lat[cnt].x = xs + j*lavg;
          lat[cnt].y = ys;	
          cnt++;
        }
        */
      }
    }
  }
  else if(strcmp(shape, "disordered_hexagon")==0){

    for(int i=0; i<Ns; i++){
      xs = (-lavg/2.0)*i;
      ys = lavg*(sqrt(3.0)/2.0)*i;
      for(int j=0; j<(Ns+i); j++){
        if(j==(Ns+i)-1 || j==0 || i==0){
          double xsi = gsl_rng_uniform(rg);
          if(xsi>0.5){
            lat[cnt].x = xs + j*lavg;
            lat[cnt].y = ys;	
            cnt++;
          }
        }
        else{
          lat[cnt].x = xs + j*lavg;
          lat[cnt].y = ys;	
          cnt++;
        }
      }
    }
    for(int i=(Ns-2); i>=0; i--){
      xs += lavg/2.0;
      ys += lavg*(sqrt(3.0)/2.0);
      for(int j=0; j<(Ns+i); j++){
        if(j==(Ns+i)-1 || j==0 || i==0){
          double xsi = gsl_rng_uniform(rg);
          if(xsi>0.5){
            lat[cnt].x = xs + j*lavg;
            lat[cnt].y = ys;	
            cnt++;
          }
        }
        else{
          lat[cnt].x = xs + j*lavg;
          lat[cnt].y = ys;	
          cnt++;
        }
      }
    }
  }
  else if(strcmp(shape, "long")==0){
    for(int i=0; i<nx; i++){
      for(int j=0; j<ny; j++){
        lat[i+j*nx].x = lavg*i;// + (lavg/2.0)*((j+1)%2);
        lat[i+j*nx].y = (sqrt(3.0)/2.0)*lavg*j;
        cnt++;
      }
    }   
  }
  else{
    printf("Shape not supported.\n");
    exit(-1);
  }
	//std::cout << cnt << "\n";
  Nlat=cnt;
  cnt = 0;
}

void initialize_bond(std::vector<site> &lat, int **diagonal, int **off_diag, arma::mat &rotkin){

	//Get nearest neighbors
	for(int i=0; i<Nlat; i++){
		int nn=0;
		for(int j=0; j<Nlat; j++){
			if(i!=j){
				if(get_dist(lat[i], lat[j])<(laa+1e-4)){
					lat[i].nbs(nn)=j;
					nn++;
				}
			}
		}
    lat[i].n = nn;
//		std::cout << nn << "\n";
	}

  //Get rotational kinetic energy matrix
  for(int i=0; i<Nlat; i++){
//    printf("neighbors: %d\n", lat[i].n);
    if(lat[i].n<6){
      rotkin(i,i) = -2;
      for(int j=0; j<lat[i].n; j++){
        int nind = (int)lat[i].nbs(j);
        if(lat[nind].n<6){
          rotkin(i,nind)=1;
        }
      }
    }
  }

  //Get bond info
  for(int i=0; i<Nlat; i++){
    diagonal[i][0] = lat[i].n;
    for(int j=0; j<lat[i].n; j++){
      int bond_index=0;
      int nb = (int)lat[i].nbs(j);
      if(lat[i].x>lat[nb].x){
        if(lat[i].y<lat[nb].y) bond_index = -3;
        if(lat[i].y==lat[nb].y) bond_index = -2;
        if(lat[i].y>lat[nb].y) bond_index = -1;
      }
      if(lat[i].x<lat[nb].x){
        if(lat[i].y<lat[nb].y) bond_index = 1;
        if(lat[i].y==lat[nb].y) bond_index = 2;
        if(lat[i].y>lat[nb].y) bond_index = 3;
      }
      off_diag[i][nb] = bond_index;
      diagonal[i][1+j] = bond_index;
    }
  }
}

/********************************************/

int is_neighbor(std::vector<site> &lat, int m1, int m2){

  int found = 0;

  for(int i=0; i<lat[m1].n; i++){
    if(m2==lat[m1].nbs(i)) found = 1;
  }

  return found;
} 

/********************************************/

double get_k(int s1, int s2){
  if(s1==1 && s2==1) return kaa;
  else if(s1==-1 && s2==-1) return kbb;
  else return kab;
}

/********************************************/

double get_l(int s1, int s2){
  if(s1==1 && s2==1) return laa;
  else if(s1==-1 && s2==-1) return lbb;
  else return lab;
}

/********************************************/

void pxyz(std::vector<site> &lat, int N, const char *name){

	FILE *f;
	f = fopen(name, "w");
	fprintf(f, "%d\n", N);
	fprintf(f, "%s\n", "comment");

	for(int i=0; i<N; i++){
		if(lat[i].s==1) fprintf(f, "Ar %f %f %f\n", lat[i].x, lat[i].y, 0.0); 
    else fprintf(f, "S %f %f %f\n", lat[i].x, lat[i].y, 0.0);
	}
	fclose(f);
	f=NULL;
}


/*********************************************/

void pquant(double *data, const char *name){

  FILE *f;
  f = fopen(name, "w");
  for(int i=0; i<(npass/nfreq); i++){
    fprintf(f, "%d %f\n", i, data[i]);
  }
  fclose(f);
  f=NULL;
}

/*********************************************/

void pbonddata(int **d, int **od){

  FILE *f;
  char outp1[200];
  if(strcmp(shape, "hexagon")==0) sprintf(outp1, "nc_matrices/stiffshell/%s/topology_diag_Ns=%d_nlayer=%d_Kshell=%f.txt", shape, Ns, nlayer, Kshell);
  if(strcmp(shape, "disordered_hexagon")==0) sprintf(outp1, "nc_matrices/%s/dis_topology_diag_Ns=%d.txt", shape, Ns);
  //sprintf(outp1, "nc_matrices/%s/topology_diag_nx=%d_ny=%d.txt", shape, nx, ny);
  f = fopen(outp1, "w");
  for(int i=0; i<Nlat; i++){
    for(int j=0; j<7; j++){
      fprintf(f, "%d ", d[i][j]);
    }
    fprintf(f, "\n");
  }
  fclose(f);
  f=NULL;

  FILE *f2;
  char outp2[200];
  if(strcmp(shape, "hexagon")==0) sprintf(outp2, "nc_matrices/stiffshell/%s/topology_offdiag_Ns=%d_nlayer=%d_Kshell=%f.txt", shape, Ns, nlayer, Kshell);
  if(strcmp(shape, "disordered_hexagon")==0) sprintf(outp2, "nc_matrices/%s/dis_topology_offdiag_Ns=%d.txt", shape, Ns);
  //sprintf(outp2, "nc_matrices/%s/topology_offdiag_nx=%d_ny=%d.txt", shape, nx, ny);
  f2 = fopen(outp2, "w");
  for(int i=0; i<Nlat; i++){
    for(int j=0; j<Nlat; j++){
      fprintf(f2, "%d ", od[i][j]);
    }
    fprintf(f2, "\n");
  }
  fclose(f2);
  f2=NULL;

}

/*********************************************/

void pmatrices(arma::mat dyn, arma::mat coup, arma::mat spin, arma::mat veff){

  FILE *f;
  char outp1[200];
  if(strcmp(shape, "hexagon")==0)  sprintf(outp1, "nc_matrices/stiffshell/%s/dyn_mat_Ns=%d_nlayer=%d_Kshell=%f.txt", shape, Ns, nlayer, Kshell);
  if(strcmp(shape, "disordered_hexagon")==0)  sprintf(outp1, "nc_matrices/%s/dis_dyn_mat_Ns=%d.txt", shape, Ns);
  //sprintf(outp1, "nc_matrices/%s/dyn_mat_nx=%d_ny=%d.txt", shape, nx, ny);
  f = fopen(outp1, "w");
  for(int i=0; i<2*Nlat-3; i++){
    for(int j=0; j<2*Nlat-3; j++){
      fprintf(f, "%lf ", dyn(i,j));
    }
    fprintf(f, "\n");
  }
  fclose(f);
  f=NULL;

  if(strcmp(shape, "hexagon")==0)  sprintf(outp1, "nc_matrices/stiffshell/%s/coup_mat_Ns=%d_nlayer=%d_Kshell=%f.txt", shape, Ns, nlayer, Kshell);
  if(strcmp(shape, "disordered_hexagon")==0)  sprintf(outp1, "nc_matrices/%s/dis_coup_mat_Ns=%d.txt", shape, Ns);
  //sprintf(outp1, "nc_matrices/%s/coup_mat_nx=%d_ny=%d.txt", shape, nx, ny);
  f = fopen(outp1, "w");
  for(int i=0; i<2*Nlat-3; i++){
    for(int j=0; j<Nlat; j++){
      fprintf(f, "%lf ", coup(i,j));
    }
    fprintf(f, "\n");
  }
  fclose(f);
  f=NULL;

  if(strcmp(shape, "hexagon")==0)  sprintf(outp1, "nc_matrices/stiffshell/%s/spin_mat_Ns=%d_nlayer=%d_Kshell=%f.txt", shape, Ns, nlayer, Kshell);
  if(strcmp(shape, "disordered_hexagon")==0)  sprintf(outp1, "nc_matrices/%s/dis_spin_mat_Ns=%d.txt", shape, Ns);
  //sprintf(outp1, "nc_matrices/%s/spin_mat_nx=%d_ny=%d.txt", shape, nx, ny);
  f = fopen(outp1, "w");
  for(int i=0; i<Nlat; i++){
    for(int j=0; j<Nlat; j++){
      fprintf(f, "%lf ", spin(i,j));
    }
    fprintf(f, "\n");
  }
  fclose(f);
  f=NULL;

  if(strcmp(shape, "hexagon")==0)  sprintf(outp1, "nc_matrices/stiffshell/%s/veff_mat_Ns=%d_nlayer=%d_Kshell=%f.txt", shape, Ns, nlayer, Kshell);
  if(strcmp(shape, "disordered_hexagon")==0)  sprintf(outp1, "nc_matrices/%s/dis_veff_mat_Ns=%d.txt", shape, Ns);
  //sprintf(outp1, "nc_matrices/%s/veff_mat_nx=%d_ny=%d.txt", shape, nx, ny);
  f = fopen(outp1, "w");
  for(int i=0; i<Nlat; i++){
    for(int j=0; j<Nlat; j++){
      fprintf(f, "%.15e ", veff(i,j));
    }
    fprintf(f, "\n");
  }
  fclose(f);
  f=NULL;

  if(strcmp(shape, "hexagon")==0)  sprintf(outp1, "nc_matrices/stiffshell/%s/veff_vec_Ns=%d_nlayer=%d_Kshell=%f.txt", shape, Ns, nlayer, Kshell);
  if(strcmp(shape, "disordered_hexagon")==0)  sprintf(outp1, "nc_matrices/%s/dis_veff_vec_Ns=%d.txt", shape, Ns);
  //sprintf(outp1, "nc_matrices/%s/veff_vec_ nx=%d_ny=%d.txt", shape, nx, ny);
  f = fopen(outp1, "w");
  for(int i=0; i<Nlat; i++){
    for(int j=0; j<Nlat; j++){
      fprintf(f, "%.15e\n", veff(i,j));
    }
  }
  fclose(f);
  f=NULL;


}

/*********************************************/

void pevals(arma::vec evals){

  FILE *f;
  char outp[300];
  if(strcmp(shape, "hexagon")==0)  sprintf(outp, "nc_matrices/stiffshell/%s/evals_Ns=%d_nlayer=%d_Kshell=%f.txt", shape, Ns, nlayer, Kshell);
  if(strcmp(shape, "disordered_hexagon")==0)  sprintf(outp, "nc_matrices/%s/dis_evals_Ns=%d.txt", shape, Ns);
  //sprintf(outp, "nc_matrices/%s/evals_nx=%d_ny=%d.txt", shape, nx, ny);
  f = fopen(outp, "w");
  for(int j=0; j<Nlat; j++){
    fprintf(f, "%d %.15e\n", j, evals(j));
  }
  fclose(f);
  f=NULL;
}

/*********************************************/

void pevecs(arma::mat evecs, std::vector<site> &lat){

  for(int i=0; i<Nlat; i++){
    FILE *f;
    char outp[300];
    if(strcmp(shape, "hexagon")==0) sprintf(outp, "nc_matrices/stiffshell/evecs/%s/evec_Ns=%d_nlayer=%d_Kshell=%f_%d.txt", shape, Ns, nlayer, Kshell, i);
    if(strcmp(shape, "disordered_hexagon")==0) sprintf(outp, "nc_matrices/evecs/%s/dis_evec_Ns=%d_%d.txt", shape, Ns, i);
//    sprintf(outp, "nc_matrices/evecs/%s/evec_nx=%d_ny=%d_%d.txt", shape, nx, ny, i);
    f = fopen(outp, "w");
    for(int j=0; j<Nlat; j++){
      fprintf(f, "%d %f %f %f\n", j, lat[j].x, lat[j].y, evecs(j,i));
    }
    fclose(f);
    f=NULL;
  }
}

/*********************************************/

void pfonevals(arma::vec evals){

  FILE *f;
  char outp[300];
  if(strcmp(shape, "hexagon")==0)  sprintf(outp, "nc_matrices/stiffshell/%s/fonevals_Ns=%d_nlayer=%d_Kshell=%f.txt", shape, Ns, nlayer, Kshell);
  if(strcmp(shape, "disordered_hexagon")==0)  sprintf(outp, "nc_matrices/%s/dis_fonevals_Ns=%d.txt", shape, Ns);
//  sprintf(outp, "nc_matrices/%s/fonevals_nx=%d_ny=%d.txt", shape, nx, ny);
  f = fopen(outp, "w");
  for(int j=0; j<2*Nlat-3; j++){
    fprintf(f, "%d %.15e\n", j, evals(j));
  }
  fclose(f);
  f=NULL;
}

/*********************************************/

void pconfigs(std::vector<site> &lat, arma::mat veff){

  for(int i=0; i<Nlat; i++){
    FILE *f;
    char outp[300];
    if(strcmp(shape, "hexagon")==0)  sprintf(outp, "nc_matrices/stffshell/pair_configs/%s/atom_%d_Ns=%d_nlayer=%d_Kshell=%f.txt", shape, i, Ns, nlayer, Kshell);
    if(strcmp(shape, "disordered_hexagon")==0)  sprintf(outp, "nc_matrices/pair_configs/%s/dis_atom_%d_Ns=%d.txt", shape, i, Ns);
//    sprintf(outp, "nc_matrices/pair_configs/%s/atom_%d_nx=%d_ny=%d.txt", shape, i, nx, ny);
    f = fopen(outp, "w");
    for(int j=0; j<Nlat; j++){
      fprintf(f, "%d %f %f %.15e\n", j, lat[j].x, lat[j].y, veff(i,j));
    }
    fclose(f);
    f=NULL;
  }
}

/*********************************************/

double get_dist(site si, site sj){

	double r = sqrt((si.x-sj.x)*(si.x-sj.x)+(si.y-sj.y)*(si.y-sj.y));

	return r;
}

/*********************************************/

double magnetization(std::vector<site> &lat){

  double d = 0;
  for(int i=0; i<Nlat; i++) d += lat[i].s;
  return d;
}

/*********************************************/

site get_com(std::vector<site> &lat){

  site com;
  com.x = 0;
  com.y = 0;

  for(int i=0; i<Nlat; i++){
    com.x += lat[i].x;
    com.y += lat[i].y;
  }

  com.x = com.x/Nlat;
  com.y = com.y/Nlat;

  return com; 
}
