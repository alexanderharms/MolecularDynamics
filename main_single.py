# Originally made in collaboration with Aloys Erkelens as an assigment
# for the course Computational Physics by J.M. Thijssen in 2015/2016.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from MD import initialize, simulate

# Simulation of argon atoms in a volume V with length L.
rho = 0.80 # Density  [kg/m3]
T = 1.462 # Temperature [120*K]. Temperature will be regulated until system is in equillibrium [120*K]
Nuc = 3 # Number of unit cells in one dimension. For actual measurements Nuc = 6 is advised

dt = 0.004 # Time step of the simulation
n_t = 5500 # Total amount of time steps. For an actual measurement n_t = 15500 is advised.
tEquilibrium = 500 # Time steps after which the temperature is constant. Condition: tEquilibrium < n_t
TimeBlock = 1000 # Time interval over which one measurement is performed

Nbins = 300 # Number of bins in histogram for the correlation function g(r)

# Define the position and velocity at t = 0
pos, vel, V, N, L, Luc, binLength = initialize(Nuc, rho, T, Nbins)

# Run the simulation for the argon atoms
posNew, velNew, potEnergy, kinEnergy, g, errorg, r, Cv, CvPre, errorCv, errorCvPre, Pressure, errorPressure = simulate(V, N, L, T, tEquilibrium, pos, vel, n_t, dt, Nbins, binLength, TimeBlock)


font = {'family' : 'serif',
        'size'   : 18}

plt.rc('font', **font)

# Plot the energy of the system
t = np.linspace(1, n_t, n_t)
fig = plt.figure()
plt.plot(t, potEnergy, label = 'Potential energy', linewidth = 2)
plt.plot(t, kinEnergy, label = 'Kinetic energy', linewidth = 2)
plt.plot(t, kinEnergy + potEnergy, label = 'Total energy', linewidth = 2)
plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
plt.ylabel('Energy')
plt.xlabel('time')
plt.title('Energy of the system')
plt.tick_params(axis = 'both', pad=10)

fig.savefig('energy.pdf', bbox_inches='tight')


# Plot the correlation function g(r)
fig2 = plt.figure()
plt.plot(r, g, linewidth = 2)
plt.axhline(y = 1, color = 'k', linestyle = 'dashed', linewidth = 2)
plt.fill_between(r, g-errorg, g+errorg, alpha = 0.5, edgecolor = '#CC4F1B', facecolor = '#FF9848')
plt.ylabel('g(r)')
plt.xlabel('r/$\sigma$')
plt.title('Correlation function')
plt.xlim(0, 0.5*L)
plt.tick_params(axis = 'both', pad = 10)

fig2.savefig('correlation.pdf', bbox_inches='tight')


# Plot the end positions of the particles
fig3 = plt.figure()
ax = fig3.add_subplot(111, projection = '3d')
ax.scatter(posNew[:, 0], posNew[:, 1], posNew[:, 2], marker = 'o')

fig3.savefig('particles.pdf', bbox_inches='tight')
