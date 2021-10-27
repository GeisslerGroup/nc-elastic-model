/* Get effective potential in bulk
 * by numerically diagonalizing matrices.
 */

//C++ headers
#include <iostream>
#include <armadillo>

//C headers
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_rng.h>

//User-defined headers
#include "elastic_ising_3d.h"
#include "site3d.h"

//Set global constants
#define J  0.0 //coupling constant
#define npass 1e5
#define nfreq 100

gsl_rng *rg; //use gsl for random number generation

int Nlat;
int nx;
int ny;
int nz;
double dx = 0.1;

double H = 0.0;
double kT = 0.1;

double laa = 1.0;
double lbb = 0.8;
double lab;
double lavg;

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

int center_index = -1;

void apply_pbc(std::vector<site3d> &lat);
void initialize_bond(std::vector<site3d> &lat, char *init, int **diagonal, int **off_diag);
int is_neighbor(std::vector<site3d> &lat, int m1, int m2);
void pbonddata(int **d, int **od);
void pmatrices(arma::mat dyn, arma::mat coup, arma::mat spin, arma::mat veff);
void pconfigs(std::vector<site3d> &lat, arma::mat veff);
void pconf_xyz(std::vector<site3d> &lat, arma::mat veff);
void print_pot(arma::mat veff);
void pevals(arma::vec evals);
void pevecs(arma::mat evecs, std::vector<site3d> &lat);
void pfonevals(arma::vec evals);
void delete_site(std::vector<site3d> &lat, int index);

/******************************************************/
/*** MAIN ***/
/******************************************************/

int main(int argc, char *argv[]){

  if(argc!=4){
    printf("Error: need nx, ny, nz.\n");
    exit(-1);
  }
  
  nx = atoi(argv[1]);
  ny = atoi(argv[2]);
  nz = atoi(argv[3]);

	//Compute global varaibles
	Nlat = nx*ny*nz;
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

  //Define FCC lattice vectors
  arma::vec a1 = {1.0, 0, 0};
  arma::vec a2 = {0, 1.0 ,0};
  arma::vec a3 = {0, 0, 1.0};

  arma::vec b1 = {1.0/sqrt(2.0), 1.0/sqrt(2.0), 0};
  arma::vec b2 = {1.0/sqrt(2.0), 0, 1.0/sqrt(2.0)};
  arma::vec b3 = {0, 1.0/sqrt(2.0), 1.0/sqrt(2.0)};
  arma::vec b4 = {0, -1.0/sqrt(2.0), 1.0/sqrt(2.0)};
  arma::vec b5 = {-1.0/sqrt(2.0), 0, 1.0/sqrt(2.0)};
  arma::vec b6 = {-1.0/sqrt(2.0), 1.0/sqrt(2.0), 0};

  arma::mat alpha_mat(9,3);
  for(int i=0; i<3; i++){
    alpha_mat(0,i) = a1(i);
    alpha_mat(1,i) = a2(i);
    alpha_mat(2,i) = a3(i);

    alpha_mat(3,i) = b1(i);
    alpha_mat(4,i) = b2(i);
    alpha_mat(5,i) = b3(i);
    alpha_mat(6,i) = b4(i);
    alpha_mat(7,i) = b5(i);
    alpha_mat(8,i) = b6(i);
  }

  //Allocate memory
	std::vector<site3d> lat(Nlat);

  int **diagonal = (int**)calloc((size_t)Nlat, sizeof(int*));
  for(int i=0; i<Nlat; i++){
    diagonal[i] = (int*)calloc((size_t)19, sizeof(int));
  }
  int **off_diag = (int**)calloc((size_t)Nlat, sizeof(int*));
  for(int i=0; i<Nlat; i++){
    off_diag[i] = (int*)calloc((size_t)Nlat, sizeof(int));
  }
  initialize_bond(lat, init, diagonal, off_diag);

  arma::mat dyn(3*Nlat, 3*Nlat, arma::fill::zeros);
  arma::mat coup(3*Nlat, Nlat, arma::fill::zeros);
  arma::mat spin(Nlat, Nlat, arma::fill::zeros); 
  arma::mat veff(Nlat, Nlat, arma::fill::zeros);
  arma::vec spinvec(Nlat, arma::fill::zeros);
  arma::vec myones(Nlat, arma::fill::zeros);
  arma::vec select(Nlat, arma::fill::zeros);
  for(int i=0; i<Nlat; i++){
    spinvec(i) = 1.0;
    myones(i) = 1.0;
  }

  //Create lattice 
  printf("initialized\n");

  /* Fill in matrices */
  for(int i=0; i<Nlat; i++){
    //Spin Matrix
    spin(i, i) = diagonal[i][0]/4.0;
    for(int j=0; j<Nlat; j++){
      if(i!=j){
        if(off_diag[i][j]!=0) spin(i,j)=1/4.0;
      }
    }
    //Dynamical Matrix
    for(int k=0; k<lat[i].n; k++){
      dyn(3*i,3*i) += alpha_mat(abs(diagonal[i][k+1])-1, 0)*alpha_mat(abs(diagonal[i][k+1])-1, 0);
      dyn(3*i,3*i+1) += alpha_mat(abs(diagonal[i][k+1])-1, 0)*alpha_mat(abs(diagonal[i][k+1])-1, 1);
      dyn(3*i+1,3*i) += alpha_mat(abs(diagonal[i][k+1])-1, 1)*alpha_mat(abs(diagonal[i][k+1])-1, 0);
      dyn(3*i+1,3*i+1) += alpha_mat(abs(diagonal[i][k+1])-1, 1)*alpha_mat(abs(diagonal[i][k+1])-1, 1);
      dyn(3*i,3*i+2) += alpha_mat(abs(diagonal[i][k+1])-1, 0)*alpha_mat(abs(diagonal[i][k+1])-1, 2);
      dyn(3*i+2,3*i) += alpha_mat(abs(diagonal[i][k+1])-1, 2)*alpha_mat(abs(diagonal[i][k+1])-1, 0);
      dyn(3*i+1,3*i+2) += alpha_mat(abs(diagonal[i][k+1])-1, 1)*alpha_mat(abs(diagonal[i][k+1])-1, 2);
      dyn(3*i+2,3*i+1) += alpha_mat(abs(diagonal[i][k+1])-1, 2)*alpha_mat(abs(diagonal[i][k+1])-1, 1);
      dyn(3*i+2,3*i+2) += alpha_mat(abs(diagonal[i][k+1])-1, 2)*alpha_mat(abs(diagonal[i][k+1])-1, 2);
    }
    for(int j=0; j<Nlat; j++){
      if(i!=j && is_neighbor(lat, i, j)){
        dyn(3*i, 3*j) = -alpha_mat(abs(off_diag[i][j])-1, 0)*alpha_mat(abs(off_diag[i][j])-1, 0);
        dyn(3*i, 3*j+1) = -alpha_mat(abs(off_diag[i][j])-1, 0)*alpha_mat(abs(off_diag[i][j])-1, 1);
        dyn(3*i+1, 3*j) = -alpha_mat(abs(off_diag[i][j])-1, 1)*alpha_mat(abs(off_diag[i][j])-1, 0);
        dyn(3*i+1, 3*j+1) = -alpha_mat(abs(off_diag[i][j])-1, 1)*alpha_mat(abs(off_diag[i][j])-1, 1);
        dyn(3*i, 3*j+2) = -alpha_mat(abs(off_diag[i][j])-1, 0)*alpha_mat(abs(off_diag[i][j])-1, 2);
        dyn(3*i+2, 3*j) = -alpha_mat(abs(off_diag[i][j])-1, 2)*alpha_mat(abs(off_diag[i][j])-1, 0);
        dyn(3*i+1, 3*j+2) = -alpha_mat(abs(off_diag[i][j])-1, 1)*alpha_mat(abs(off_diag[i][j])-1, 2);
        dyn(3*i+2, 3*j+1) = -alpha_mat(abs(off_diag[i][j])-1, 2)*alpha_mat(abs(off_diag[i][j])-1, 1);
        dyn(3*i+2, 3*j+2) = -alpha_mat(abs(off_diag[i][j])-1, 2)*alpha_mat(abs(off_diag[i][j])-1, 2);
      }
    }
    //Coupling Matrix
    for(int k=0; k<lat[i].n; k++){
      coup(3*i, i) += alpha_mat(abs(diagonal[i][k+1])-1, 0)*copysign(1.0, -diagonal[i][k+1]);
      coup(3*i+1, i) += alpha_mat(abs(diagonal[i][k+1])-1, 1)*copysign(1.0, -diagonal[i][k+1]);
      coup(3*i+2, i) += alpha_mat(abs(diagonal[i][k+1])-1, 2)*copysign(1.0, -diagonal[i][k+1]);
    }
    for(int j=0; j<Nlat; j++){
      if(i!=j && is_neighbor(lat, i, j)){
        coup(3*i, j) += alpha_mat(abs(off_diag[i][j])-1, 0)*copysign(1.0, -off_diag[i][j]);
        coup(3*i+1, j) += alpha_mat(abs(off_diag[i][j])-1, 1)*copysign(1.0, -off_diag[i][j]);
        coup(3*i+2, j) += alpha_mat(abs(off_diag[i][j])-1, 2)*copysign(1.0, -off_diag[i][j]);
      }
    }
  }

  /* Construct V_eff. Need to remove zero modes in NC (not bulk.) */


 
  dyn.shed_row(3*nx*ny*nz-2);
  dyn.shed_col(3*nx*ny*nz-2);
  dyn.shed_row(3*nx*ny-1);
  dyn.shed_col(3*nx*ny-1);
  dyn.shed_row(3*(nx*(ny-1)+1)-3);
  dyn.shed_col(3*(nx*(ny-1)+1)-3);
/*  dyn.shed_row(3*nx-2);
  dyn.shed_col(3*nx-2);
 */ 
  dyn.shed_rows(0,2);
  dyn.shed_cols(0,2);

  coup.shed_row(3*nx*ny*nz-2);
  coup.shed_row(3*nx*ny-1);
  coup.shed_row(3*(nx*(ny-1)+1)-3);
//  coup.shed_row(3*nx-2);
  coup.shed_rows(0,2);

  veff = 4.0*spin - coup.t()*inv(dyn)*coup;

  printf("Obtained effective potential\n");

  spinvec(0) = -1;
  //spinvec(Nlat/2) = -1;

  double energy = dot((spinvec-myones*mean(spinvec)), veff*(spinvec-myones*mean(spinvec)));
  printf("energy: %f\n", energy);
  double energy2 = dot((spinvec), veff*(spinvec));
  printf("energy2: %f\n", energy2);

  select(1) = 1;
  double test = dot(select, veff*myones);
  printf("test %f\n", test);

  arma::vec eigval;
  arma::mat eigvec = arma::zeros(Nlat-3, Nlat-3);
  eig_sym(eigval, eigvec, veff);//+3.14159*rotkin);
  for(int i=0; i<Nlat; i++) printf("eval %d: %.15e\n", i, eigval(i));
  for(int i=0; i<Nlat; i++) spinvec(i) = -1.0/sqrt(Nlat);
  //std::cout << eigvec.col(0) << std::endl;
//  for(int i=0; i<Nlat; i++) printf("orthogonal? %f\n", dot(eigvec.col(1), eigvec.col(i)));

  int num_surf=0;
  int num_bond = 0;
  for(int i=0; i<Nlat; i++){
    if(lat[i].n<18) num_surf++;
    num_bond += lat[i].n;
  }
  num_bond = num_bond/2;
  printf("num surf: %d\n", num_surf);
  printf("Nlat: %d\n", Nlat);
  printf("num bonds: %d\n", num_bond);
  //printf("rule count: %d\n", (3+1)*(Nlat-1)-num_bond);

  //Phonon Matrix
  /*
  arma::vec phonon_eval;
  arma::mat phonon_evec = arma::zeros(3*Nlat-3, 3*Nlat-3);
  printf("phonon matrix\n");
  eig_sym(phonon_eval, phonon_evec, dyn);

  std::cout << phonon_eval << std::endl;
//  std::cout << veff << std::endl;
  */
  pmatrices(dyn, coup, spin, veff);
	//pbonddata(diagonal, off_diag);
  //pevals(eigval);
  //pevecs(eigvec, lat);
  //pfonevals(phonon_eval);
  //pconfigs(lat, veff);
  //pconf_xyz(lat, veff);
  //print_pot(veff);
}

/* Function Definitions *********************************************************************************/

void initialize_bond(std::vector<site3d> &lat, char *init, int **diagonal, int **off_diag){

	//First Set spin values
  if(strcmp(init, "ones")==0){
    for(int i = 0; i < Nlat; i++) lat[i].s = 1;
  }
  else{
    printf("Not a valid initial configuration.\n");
    exit(-1);
  }

	//Next set position values
  
  lavg = laa;
  double a_lat = lavg;

  int cnt = 0;

  for(int k=0; k<nz; k++){
    for(int j=0; j<ny; j++){
      for(int i=0; i<nx; i++){

        lat[cnt].x = a_lat*i;
        lat[cnt].y = a_lat*j;
        lat[cnt].z = a_lat*k;
        lat[cnt].sublat=1;

        cnt+=1;
      }
    }
  }

  //Center the lattice at 0
  
  site3d com1 = get_com(lat);
  double dist1 = 100;
  //Get index of atom closest to com
  for(int i=0; i<Nlat; i++){
    double tempdist = get_dist(lat[i],com1);
    if(tempdist<dist1){
      center_index = i;
      dist1 = tempdist;
    }
  }
  double deltax = lat[center_index].x;
  double deltay = lat[center_index].y;
  double deltaz = lat[center_index].z;

  for(int i=0; i<Nlat; i++){
    lat[i].x -= deltax;
    lat[i].y -= deltay;
    lat[i].z -= deltaz;
  }

  //Carve out a sphere
 /* 
  int done = 0;
  double maxz = 4.0;
  while(done==0){
    for(int i=0; i<Nlat; i++){
      if(sqrt(lat[i].x*lat[i].x+lat[i].y*lat[i].y+lat[i].z*lat[i].z)>(maxz+1e-3)){
        delete_site(lat, i);
        break;
      }
      if(i==(Nlat-1)) done=1;
    }
  }
  printf("made a sphere\n");
 */

	//Get nearest neighbors
	for(int i=0; i<Nlat; i++){
		int nn=0;
		for(int j=0; j<Nlat; j++){
			if(i!=j){
				if(get_dist(lat[i], lat[j])<(sqrt(2.0)*laa+1e-4)){
					lat[i].nbs(nn)=j;
					nn++;
				}
			}
		}
    lat[i].n = nn;
		//std::cout << nn << "\n";
	}


  //Get bond info
  //Need to re-do for 3d
  int numtotbond=0;
  for(int i=0; i<Nlat; i++){
    diagonal[i][0] = lat[i].n;
    for(int j=0; j<lat[i].n; j++){
      numtotbond++;
      int bond_index=0;
      int nb = (int)lat[i].nbs(j);
      //minimum image convention
      double deex = lat[i].x-lat[nb].x;
      //if(deex>Lx/2.0) deex = deex - Lx;
      //if(deex<=-Lx/2.0) deex = deex + Lx;
      double deey = lat[i].y-lat[nb].y;
      //if(deey>Ly/2.0) deey = deey - Ly;
      //if(deey<=-Ly/2.0) deey = deey + Ly;
      double deez = lat[i].z-lat[nb].z;
      //if(deez>Lz/2.0) deez = deez - Lz;
      //if(deez<=-Lz/2.0) deez = deez + Lz;

      if(deez>0){
        if(deex>0){
          bond_index = 5;
        }
        if(deex<0){
          bond_index = 8;
        }
        if(deex==0){
          if(deey>0){
            bond_index = 6;
          }
          if(deey<0){
            bond_index = 7;
          }
          if(deey==0){
            bond_index = 3;
          }
        }
      }
      if(deez==0){
        if(deey>0){
          if(deex>0){
            bond_index = 4;
          }
          if(deex<0){
            bond_index = 9;
          }
          if(deex==0){
            bond_index = 2;
          }
        }
        if(deey<0){
          if(deex>0){
            bond_index = -9;
          }
          if(deex<0){
            bond_index = -4;
          }
          if(deex==0){
            bond_index = -2;
          }
        }
        if(deey==0){
          if(deex>0) bond_index = 1;
          if(deex<0) bond_index = -1;
        }

      }
      if(deez<0){
        if(deex>0){
          bond_index = -8;
        }
        if(deex<0){
          bond_index = -5;
        }
        if(deex==0){
          if(deey>0){
            bond_index = -7;
          }
          if(deey<0){
            bond_index = -6;
          }
          if(deey==0){
            bond_index = -3;
          }
        }
      }
      off_diag[i][nb] = bond_index;
      diagonal[i][1+j] = bond_index;
    }
  }
  printf("num bonds: %d\n", numtotbond);
}

/********************************************/

int is_neighbor(std::vector<site3d> &lat, int m1, int m2){

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

void pxyz(std::vector<site3d> &lat, int N, const char *name){

	FILE *f;
	f = fopen(name, "w");
	fprintf(f, "%d\n", N);
	fprintf(f, "%s\n", "comment");

	for(int i=0; i<N; i++){
		if(lat[i].s==1) fprintf(f, "Ar %f %f %f\n", lat[i].x, lat[i].y, lat[i].z); 
    else fprintf(f, "S %f %f %f\n", lat[i].x, lat[i].y, lat[i].z);
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
  sprintf(outp1, "nc_matrices/3d/cube/topology_diag_nx=%d_ny=%d_nz=%d.txt", nx, ny, nz);
  f = fopen(outp1, "w");
  for(int i=0; i<Nlat; i++){
    for(int j=0; j<13; j++){
      fprintf(f, "%d ", d[i][j]);
    }
    fprintf(f, "\n");
  }
  fclose(f);
  f=NULL;

  FILE *f2;
  char outp2[200];
  sprintf(outp2, "nc_matrices/3d/cube/topology_offdiag_nx=%d_ny=%d_nz=%d.txt", nx, ny, nz);
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

  //arma::mat conv_mat(3*Nlat, Nlat, arma::fill::zeros);
  //conv_mat = inv(dyn)*coup;

  FILE *f;
  char outp1[200];
  /*
  sprintf(outp1, "nc_matrices/3d/cube/dyn_mat_nx=%d_ny=%d_nz=%d.txt", nx, ny, nz);
  f = fopen(outp1, "w");
  for(int i=0; i<3*Nlat-6; i++){
    for(int j=0; j<3*Nlat-6; j++){
      fprintf(f, "%lf ", dyn(i,j));
    }
    fprintf(f, "\n");
  }
  fclose(f);
  f=NULL;

  sprintf(outp1, "nc_matrices/3d/cube/coup_mat_nx=%d_ny=%d_nz=%d.txt", nx, ny, nz);
  f = fopen(outp1, "w");
  for(int i=0; i<3*Nlat-6; i++){
    for(int j=0; j<Nlat; j++){
      fprintf(f, "%lf ", coup(i,j));
    }
    fprintf(f, "\n");
  }
  fclose(f);
  f=NULL;

  sprintf(outp1, "nc_matrices/3d/cube/spin_mat_nx=%d_ny=%d_nz=%d.txt", nx, ny, nz);
  f = fopen(outp1, "w");
  for(int i=0; i<Nlat; i++){
    for(int j=0; j<Nlat; j++){
      fprintf(f, "%lf ", spin(i,j));
    }
    fprintf(f, "\n");
  }
  fclose(f);
  f=NULL;
  */
  sprintf(outp1, "nc_matrices/cube/veff_vec_nx=%d_ny=%d_nz=%d.txt", nx, ny, nz);
  f = fopen(outp1, "w");
  for(int i=0; i<Nlat; i++){
    for(int j=0; j<Nlat; j++){
      fprintf(f, "%.15e\n", veff(i,j));
    }
  }
  fclose(f);
  f=NULL;
  /*
  sprintf(outp1, "nc_matrices/3d/cube/conv_mat_nx=%d_ny=%d_nz=%d.txt", nx, ny, nz);
  f = fopen(outp1, "w");
  for(int i=0; i<3*Nlat-6; i++){
    for(int j=0; j<Nlat; j++){
      fprintf(f, "%lf ", conv_mat(i,j));
    }
    fprintf(f, "\n");
  }
  fclose(f);
  f=NULL;
  */
}

/*********************************************/

void pevals(arma::vec evals){

  FILE *f;
  char outp[300];
  sprintf(outp, "nc_matrices/cube/evals_nx=%d_ny=%d_nz=%d.txt", nx, ny, nz);
  f = fopen(outp, "w");
  for(int j=0; j<Nlat; j++){
    fprintf(f, "%d %.15e\n", j, evals(j));
  }
  fclose(f);
  f=NULL;
}

/*********************************************/

void pevecs(arma::mat evecs, std::vector<site3d> &lat){

  for(int i=0; i<Nlat; i++){
    FILE *f;
    char outp[300];
    sprintf(outp, "nc_matrices/3d/cube/evecs/evec_nx=%d_ny=%d_nz=%d_%d.txt", nx, ny, nz, i);
    f = fopen(outp, "w");
    for(int j=0; j<Nlat; j++){
      fprintf(f, "%d %f %f %f %f\n", j, lat[j].x, lat[j].y, lat[j].z, evecs(j,i));
    }
    fclose(f);
    f=NULL;
  }
}

/*********************************************/

void pfonevals(arma::vec evals){

  FILE *f;
  char outp[300];
  sprintf(outp, "nc_matrices/3d/cube/fonevals_nx=%d_ny=%d_nz=%d.txt", nx, ny, nz);
  f = fopen(outp, "w");
  for(int j=0; j<2*Nlat-3; j++){
    fprintf(f, "%d %.15e\n", j, evals(j));
  }
  fclose(f);
  f=NULL;
}


/*********************************************/

void pconfigs(std::vector<site3d> &lat, arma::mat veff){

  for(int i=0; i<Nlat; i++){
    FILE *f;
    char outp[300];
    sprintf(outp, "nc_matrices/3d/cube/pair_configs/atom_%d_nx=%d_ny=%d_nz=%d.txt", i, nx, ny, nz);
    f = fopen(outp, "w");
    for(int j=0; j<Nlat; j++){
      fprintf(f, "%d %f %f %f %.15e\n", j, lat[j].x, lat[j].y, lat[j].z, veff(i,j));
    }
    fclose(f);
    f=NULL;
  }
}

/********************************************/

void pconf_xyz(std::vector<site3d> &lat, arma::mat veff){

  for(int i=0; i<Nlat; i++){
    FILE *f;
    char outp[300];
    sprintf(outp, "nc_matrices/3d/cube/pair_configs/atom_%d_nx=%d_ny=%d_nz=%d.xyz", i, nx, ny, nz);
    f = fopen(outp, "w");
  	fprintf(f, "%d\n", Nlat);
	  fprintf(f, "%s\n", "comment");

    for(int j=0; j<Nlat; j++){
      fprintf(f, "Ar %f %f %f %.15e\n", lat[j].x, lat[j].y, lat[j].z, veff(i,j)-24.0/Nlat);
    }
    fclose(f);
    f=NULL;
  }
}

/*********************************************/

void print_pot(arma::mat veff){

  for(int i=0; i<Nlat; i++){
    FILE *f;
    char outp[300];
    sprintf(outp, "nc_matrices/3d/cube/pair_configs/potential_%d_nx=%d_ny=%d_nz=%d.xyz", i, nx, ny, nz);
    f = fopen(outp, "w");
  	fprintf(f, "%d\n", Nlat);
	  fprintf(f, "%s\n", "comment");

    for(int j=0; j<Nlat; j++){
      fprintf(f, "%f ", veff(i,j)-24.0/Nlat); //change?
    }
    fclose(f);
    f=NULL;

  }
}


/*********************************************/

double get_dist(site3d si, site3d sj){

  double x0 = si.x-sj.x;
  double y0 = si.y-sj.y;
  double z0 = si.z-sj.z;

	double r = sqrt(x0*x0+y0*y0+z0*z0);

	return r;
}

/*********************************************/

double magnetization(std::vector<site3d> &lat){

  double d = 0;
  for(int i=0; i<Nlat; i++) d += lat[i].s;
  return d;
}

/*********************************************/

site3d get_com(std::vector<site3d> &lat){

  site3d com;
  com.x = 0;
  com.y = 0;
  com.z = 0;

  for(int i=0; i<Nlat; i++){
    com.x += lat[i].x;
    com.y += lat[i].y;
    com.z += lat[i].z;
  }

  com.x = com.x/Nlat;
  com.y = com.y/Nlat;
  com.z = com.z/Nlat;

  return com; 
}

/*********************************************/

void delete_site(std::vector<site3d> &lat, int index){

  for(int i=index; i<(Nlat-1); i++){
    lat[i].x = lat[i+1].x;
    lat[i].y = lat[i+1].y;
    lat[i].z = lat[i+1].z;
  }
  Nlat--;

}
