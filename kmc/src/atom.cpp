#include "atom.h"

atom::atom() : pos(3){

	index =  -1;
	reset_pos();
}

void atom::reset_pos() {

	pos(0) = 0.0;
	pos(1) = 0.0;
	pos(2) = 0.0;
}

void atom::set_params(double x,
		double y, double z, int sval,
		int nval, int indexval){

	pos(0) = x;
	pos(1) = y;
	pos(2) = z;
	s = sval;
	n = nval;
	index = indexval;
}
