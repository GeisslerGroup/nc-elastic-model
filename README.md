# nc-elastic-model
This library provides code for running and analyzing elastic model simulations of cation exchange.

The elastic model is described in our manuscript, "Elastic Forces Drive Nonequilibrium Pattern Formation in a Model of Nanocrystal Ion Exchange," a preprint of which can be found at:

https://arxiv.org/abs/2103.05685

The code is grouped into three modules: "effective potential" for computing effective elastic interaction potentials, "kmc" for performing kinetic Monte Carlo (KMC) simulations, and "coreshell" for running equilibrium Monte Carlo simulations of core/shell nanocrystals. C++ code for running simulations of the model, along with makefiles for compiling the code, is located in the "src" subdirectories. Python scripts for performing analysis of the model trajectories are located in the "scripts" subdirectories.
Note that the C++ code requires C++11 and the Armadillo library, and that the Python scripts require Python 3 as well as NumPy and SciPy.

This software is distributed under the GPL-3 license and you are free to adapt, modify, and reuse it in other free software.
