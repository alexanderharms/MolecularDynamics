import numpy as np
import matplotlib.pyplot as plt

from moleculardynamics.base import Environment, ParticlesLJ
from moleculardynamics.vis import plot_energy, plot_g
from moleculardynamics.measurements import calc_heat_capacity
from moleculardynamics.measurements import calc_histogram, calc_g

# Simulation of argon atoms in a volume V with length L.
num_unit_cells_x = 3 # Number of unit cells in one dimension.
num_particles = 4 * num_unit_cells_x**3
rho = 0.80 # Density  [kg/m3]
dimens = [(num_particles/rho) ** (1.0/3.0)] * 3
# Temperature [120*K]. 
# Temperature will be regulated until system is in equillibrium [120*K]
temp = 1.462 

dt = 0.004 # Time step of the simulation
num_steps = 5500 # Total amount of time steps.
# Time steps after which the temperature is constant. Condition: tEquilibrium < n_t
t_equilibrium = 500 
t_interval = 10

block_size = 1000 # Time interval over which one measurement is performed
num_blocks = int((num_steps - t_equilibrium)/block_size)
num_bins = 300 # Number of bins in histogram for the correlation function g(r)

envir = Environment(dimens, temp)
envir.init_thermostat(t_equilibrium, t_interval)
particles = ParticlesLJ(num_particles, envir)

kin_energy = np.zeros(num_steps)
pot_energy = np.zeros(num_steps)
hist_array = np.zeros((num_steps, num_bins))

for step in range(num_steps):
    if step % 100 == 0:
        print("Step {} of {}".format(step, num_steps))
    particles.take_step(dt)
    kin_energy[step] = particles.kin_energy
    pot_energy[step] = particles.pot_energy
    hist_array[step, :] = calc_histogram(particles.pos, envir.dimens, num_bins)

plot_energy(num_steps, dt, kin_energy, pot_energy)
cv, _ = calc_heat_capacity(kin_energy, particles.num_particles, 
        block_size, num_blocks)

# Calculate g
# Delete timesteps before equilibrium
hist_array = np.delete(hist_array, np.s_[0:t_equilibrium], axis = 0) 
g_mean, g_error, r = calc_g(hist_array, particles.num_particles,
        envir.dimens, num_bins, block_size, num_blocks)

plot_g(r, g_mean, g_error, envir.dimens[0])


