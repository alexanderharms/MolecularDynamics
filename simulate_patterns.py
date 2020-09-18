from abc import ABC
import numpy as np

from md.particles import Particles
from md.environment import Space, Thermostat
from md.measurements import KinEnergy_Meas, TotEnergy_Meas

class Simulation(): 
    # Controls the simulation process
    def __init__(self, dt, num_steps, particles, space, boundaries, 
                 measurements):
        self.dt = dt
        self.num_steps = num_steps
        self.time_steps = 0
        self.time = 0
        self.particles = particles
        self.space = space
        self.boundaries = boundaries
        self.measurements = measurements
        self.__init_simulation()
 
    def __init_simulation(self):
        self.space.initialize(self)
        self.particles.initialize(self)
        for boundary in self.boundaries:
            boundary.initialize(self)
        for measurement in self.measurements:
            measurement.initialize(self)

    def __step_particles(self):
        pos = self.particles.positions
        vel = self.particles.velocities
        # Move the particle according to the Verlet algorithm
        vel += 0.5 * self.particles.forces * self.dt
        pos += vel * self.dt
        self.space.apply_bnd_conds(self)

        self.particles.calc_forces(self)
        vel += 0.5 * self.particles.forces * self.dt
        self.particles.positions = pos
        self.particles.velocities = vel

    def run(self):
        for step in range(num_steps):
            if step % (num_steps//100) == 0:
                print("Step {} of {}".format(step, num_steps))
            self.__step_particles()
            self.__apply_bnd_conds()
            self.__measure()
            self.time_steps += 1
            self.time = self.time_steps * self.dt

    def __apply_bnd_conds(self):
        self.space.apply_bnd_conds(self)
        for boundary in self.boundaries:
            boundary.apply_bnd_conds(self)

    def __measure(self):
        for measurement in self.measurements:
            measurement.measure(self)

    def visualize_measurements(self):
        for measurement in self.measurements:
            measurement.visualize()

if __name__ == "__main__":
    num_particles = 108
    dt = 0.004 # Time step of the simulation
    num_steps = 5500 # Total amount of time steps.
    temp = 1.462 # Initial temperature
    density = 0.80 # Initial density
    dimensions = [(num_particles/density) ** (1.0/3.0)] * 3

    therm_temp = temp
    therm_eq_time = 500
    therm_interval = 10
    space = Space(dimensions, temp)
    particles = Particles(num_particles)
    boundaries = [Thermostat(temperature=therm_temp,
                             eq_time=therm_eq_time,
                             interval=therm_interval)]
    measurements = [KinEnergy_Meas(), TotEnergy_Meas()]

    sim = Simulation(dt=dt, num_steps=num_steps, particles=particles,
                     space=space, boundaries=boundaries,
                     measurements=measurements)
    sim.run()
    sim.visualize_measurements()
