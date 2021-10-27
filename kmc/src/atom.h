#ifndef ATOM_H
#define ATOM_H

#include <armadillo>

class atom{

	public:
		atom();

		int n;
		int index;
		int s;

		arma::vec pos;
		std::vector<int> nbs;

		void reset_pos();
		void reset_nbs();
		void set_params(double x, double y, 
				double z, int sval, int nval,
				int indexval);

		~atom(){};
};

#endif
