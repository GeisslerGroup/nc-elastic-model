#include "atom.h"
#include "mc.h"

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

  return exp(-(E2-E1)/kT);

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
  lat[ind1].s = -lat[ind1].s;
  lat[ind2].s = -lat[ind2].s;
  double E2 = 0;
  for(int j=0; j<lat[ind1].n; j++){
    if(lat[ind1].nbs[j] != ind2) E2 += (-J)*lat[ind1].s*lat[lat[ind1].nbs[j]].s;
  }
  for(int j=0; j<lat[ind2].n; j++){
    if(lat[ind2].nbs[j] != ind1) E2 += (-J)*lat[ind2].s*lat[lat[ind2].nbs[j]].s;
  }
  E2 += get_elastic_swap(lat, veff, ind1, ind2);
  lat[ind1].s = -lat[ind1].s;
  lat[ind2].s = -lat[ind2].s;

  return exp(-(E2-E1)/kT);
}
