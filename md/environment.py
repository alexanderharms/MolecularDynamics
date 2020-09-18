from abc import ABC
import numpy as np

class Space():
    # Contain information on the physical space that the particles are in.
    # Currently only a cubic box
    def __init__(self, dimensions, temperature):
        self.dimensions = dimensions
        self.temperature = temperature

    def initialize(self, sim):
        pass

    def apply_bnd_conds(self, sim):
        sim.particles.positions = sim.particles.positions \
                % self.dimensions[0]

class Boundary(ABC):
    def initialize(self, sim):
        pass

    def apply_bnd_conds(self, sim):
        pass

class Thermostat(Boundary):
    def __init__(self, temperature, eq_time, interval):
        self.temperature = temperature
        self.eq_time = eq_time 
        self.interval = interval

    def apply_bnd_conds(self, sim):
        # Adjust velocity to acquire a constant temperature TDesired.
        # After tEquilibrium system should be in equilibrium with a 
        # somewhat constant temperature
        if sim.time_steps < self.eq_time \
                and (sim.time_steps % self.interval == 0):
            vel = sim.particles.velocities
            kin_energy = 0.5 * np.sum(vel * vel) 
            num_particles = sim.particles.num_particles
            vel = vel * np.sqrt(
                          (num_particles-1)*3*self.temperature 
                          / (2*kin_energy))
            sim.particles.velocities = vel
