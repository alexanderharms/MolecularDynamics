# Originally made in collaboration with Aloys Erkelens as an assigment
# for the course Computational Physics by J.M. Thijssen in 2015/2016.

import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from MD import MolecularDynamics

# Simulation of argon atoms in a volume V with length L.
rho = 1 # Density  [kg/m3]
T = 1.462 # Temperature [120*K]. Temperature will be regulated until system is in equillibrium [120*K]
Nuc = 3 # Number of unit cells in one dimension.

dt = 0.004 # Time step of the simulation
n_t = 5500 # Total amount of time steps.
t_equilibrium = 500 # Time steps after which the temperature is constant. Condition: tEquilibrium < n_t
time_block = 1000 # Time interval over which one measurement is performed

Nbins = 300 # Number of bins in histogram for the correlation function g(r)

# Instantiate the class
MDInst = MolecularDynamics(dt, n_t, t_equilibrium, Nuc, Nbins, time_block)
# Define the position and velocity at t = 0
MDInst.initialize(rho, T)
# Run the simulation for the argon atoms
MDInst.simulate()

# # Plot the end positions of the particles
# fig3 = plt.figure()
# ax = fig3.add_subplot(111, projection = '3d')
# ax.scatter(posNew[:, 0], posNew[:, 1], posNew[:, 2], marker = 'o')
#
# fig3.savefig('particles.pdf', bbox_inches='tight')
