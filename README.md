# WeightedVMC
source code for the manuscript: Improved energies and wave function accuracy with Weighted Variational Monte Carlo 

Codes for solving the ground state wavefunction:

ClassicalVMC.py: Traditional VMC code (stochastic reconfigurations) for 1d XXZ model
weightedVMC_MT.py: Code for weighted VMC using mixed tempering distribution.
weightedVMC_WTMD.py: Code for weighted VMC using well-tempered metadynamics distribution
weightedVMC_testEnergy_general.py: sampling the static parameterized wavefunction density to estimate the energy for WTMD/Classical VMC
weightedVMC_testEnergy_MT.py: sampling the static parameterized wavefunction density to estimate the energy for MT

Code for plotting the figures (Fig 2,4,6 are trivial box plots):

Hloc_utils.py    
EMUS_func_utils.py   
Fig1_tempering_projction.ipynb
Fig3_weighted_vmc_Hloc.ipynb 
Fig5_incorporate_physics_version2.ipynb 
Fig7_alpha_test.ipynb     


