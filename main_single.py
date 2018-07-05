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

# Instantiate the simulation
SimInst = MolecularDynamics(dt, n_t, t_equilibrium, Nuc, Nbins, time_block, rho, T)
# Run the simulation for the argon atoms
SimInst.simulate()

print('The number of measurements is: ', SimInst.number_blocks, '\n')

print('Heat Capacity: ',  round(SimInst.Cv, 4), '+-',  round(SimInst.errorCv, 4))
print('Prefactor: ',  round(SimInst.CvPre, 4), '+-', round(SimInst.errorCvPre, 4))
print('Pressure: ',  round(SimInst.meanPressure, 4), '+-', round(SimInst.errorPressure, 4))
print('Average Temperature: ',  round(np.average(SimInst.T_actual[t_equilibrium:]), 4))

font = {'family' : 'serif', 'size'   : 18}

plt.rc('font', **font)

# Plot the energy of the system
t = np.linspace(1, n_t, n_t)
fig = plt.figure()
plt.plot(t, SimInst.pot_energy, label = 'Potential energy', linewidth = 2)
plt.plot(t, SimInst.kin_energy, label = 'Kinetic energy', linewidth = 2)
plt.plot(t, SimInst.kin_energy + SimInst.pot_energy, label = 'Total energy', linewidth = 2)
plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
plt.ylabel('Energy')
plt.xlabel('time')
plt.title('Energy of the system')
plt.tick_params(axis = 'both', pad=10)

fig.savefig('energy.pdf', bbox_inches='tight')

# Plot the correlation function g(r)
fig2 = plt.figure()
plt.plot(SimInst.r, SimInst.g, linewidth = 2)
plt.axhline(y = 1, color = 'k', linestyle = 'dashed', linewidth = 2)
plt.fill_between(SimInst.r, SimInst.g-SimInst.errorg, SimInst.g+SimInst.errorg,
    alpha = 0.5, edgecolor = '#CC4F1B', facecolor = '#FF9848')
plt.ylabel('g(r)')
plt.xlabel('r/$\sigma$')
plt.title('Correlation function')
plt.xlim(0, 0.5*SimInst.L)
plt.tick_params(axis = 'both', pad = 10)

fig2.savefig('correlation.pdf', bbox_inches='tight')


# # Plot the end positions of the particles
# fig3 = plt.figure()
# ax = fig3.add_subplot(111, projection = '3d')
# ax.scatter(posNew[:, 0], posNew[:, 1], posNew[:, 2], marker = 'o')
#
# fig3.savefig('particles.pdf', bbox_inches='tight')
