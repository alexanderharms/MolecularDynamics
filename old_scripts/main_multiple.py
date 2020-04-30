# Originally made in collaboration with Aloys Erkelens as an assigment
# for the course Computational Physics by J.M. Thijssen in 2015/2016.

import matplotlib.pyplot as plt
import numpy as np
from MD import initialize, simulate

# Simulation of argon in a volume V with length L.
# This version of the simulation allows for a sweep along multiple densities at a fixed temperature.

dt = 0.004 # Time step of the simulation
n_t = 5500 # Total amount of time steps
tEquilibrium = 500 # Time steps after which the temperature is constant. Condition: tEquilibrium < n_t
TimeBlock = 1000 # time interval over which one measurement is performed

Nbins = 10 # number of bins in histogram for the correlation function g(r)

T = 1.36 # Temperature. Epsilon/k_b [120*K]
rho_step = np.linspace(0.10, 0.80, 8) # Density  [kg/m3]
Nuc = 3 # Number of unit cells in one dimension. For an actual measurement Nuc = 4 or higher is advised


Cv = np.zeros(len(rho_step))
CvPre = np.zeros(len(rho_step))
ErrorCv = np.zeros(len(rho_step))
ErrorCvPre = np.zeros(len(rho_step))
Pressure = np.zeros(len(rho_step))
errorPressure = np.zeros(len(rho_step))

for ii in range(len(rho_step)):
    # Define the position and velocity at t = 0
    pos, vel, V, N, L, Luc, binLength = initialize(Nuc, rho_step[ii], T, Nbins)

    print('density: ', rho_step[ii])

    # Run the simulation for different rho
    posNew, velNew, potEnergy, kinEnergy, g, errorg, r, Cv[ii], CvPre[ii], ErrorCv[ii], ErrorCvPre[ii], Pressure[ii], errorPressure[ii] = simulate(V, N, L, T, tEquilibrium, pos, vel, n_t, dt, Nbins, binLength, TimeBlock)

# Plot P*beta as a function of rho for an isotherm

font = {'family' : 'serif',
       'size'   : 18}

plt.rc('font', **font)

fig = plt.figure()
plt.plot(rho_step, Pressure*rho_step)
plt.fill_between(rho_step, rho_step*(Pressure-errorPressure), rho_step*(Pressure+errorPressure), alpha = 0.5, edgecolor = '#CC4F1B', facecolor = '#FF9848')
plt.ylabel(r'P$\beta$')
plt.xlabel(r'$\rho$')
plt.ylim(0, 2)
plt.tick_params(axis = 'both', pad = 10)
fig.savefig('pressure.pdf', bbox_inches='tight')
