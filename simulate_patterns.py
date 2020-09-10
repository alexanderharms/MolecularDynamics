from abc import ABC
import numpy as np
from numba import jit

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

    def __step(self):
        pass

    def run(self):
        for k in num_steps:
            self.step()

    def __apply_bnd_conds(self):
        pass

    def __measure(self):
        pass

class Particles():
    # Controls the particles
    def __init__(self, num_particles):
        self.num_particles = num_particles
        self.positions = np.zeros((num_particles, 3))
        self.velocities = np.zeros((num_particles, 3))
        self.forces = np.zeros((num_particles, 3))

    def initialize(self, sim):
        self.__init_positions(sim)
        self.__init_velocities(sim)

    def __init_positions(self, sim):
        # The volume is assumed to be cubic; only the first argument of
        # self.envir.dimens is used.
        dimens = [sim.space.dimensions[0]] * len(sim.space.dimensions)

        # the initial position is packed fcc.
        # 4 particles per fcc unit cell; number of unit cells in each dimension
        # should be equal.
        num_uc_onedim = round((self.num_particles//4)**(1.0/3.0))
        num_uc = num_uc_onedim ** 3
        self.num_particles = num_uc * 4
        print('the number of particles in the simulation is:', self.num_particles)

        length_uc = dimens[0] / num_uc_onedim

        # determines the initial positions for the atoms in a fcc structure.
        # position vector in one dimension
        pos_vec = np.linspace(0, self.envir.dimens[0] - length_uc, num_uc_onedim)

        # square meshgrid of positions for the atoms
        pos_mat_x, pos_mat_y, pos_mat_z = np.meshgrid(pos_vec, pos_vec, pos_vec)
        # reshape meshgrid into 1d arrays
        pos_vec_x = np.reshape(pos_mat_x, (num_uc, 1))
        pos_vec_y = np.reshape(pos_mat_y, (num_uc, 1))
        pos_vec_z = np.reshape(pos_mat_z, (num_uc, 1))

        # shape position vectors for the first set of particles on the square grid
        pos_atom_1 = np.hstack((pos_vec_x, pos_vec_y, pos_vec_z))

        pos_atom_2 = np.zeros((num_uc, 3))
        pos_atom_2[:, 0] = pos_atom_1[:, 0] + length_uc/2.0
        pos_atom_2[:, 1] = pos_atom_1[:, 1] + length_uc/2.0
        pos_atom_2[:, 2] = pos_atom_1[:, 2]

        pos_atom_3 = np.zeros((num_uc, 3))
        pos_atom_3[:, 0] = pos_atom_1[:, 0]
        pos_atom_3[:, 1] = pos_atom_1[:, 1] + length_uc/2.0
        pos_atom_3[:, 2] = pos_atom_1[:, 2] + length_uc/2.0

        pos_atom_4 = np.zeros((num_uc, 3))
        pos_atom_4[:, 0] = pos_atom_1[:, 0] + length_uc/2.0
        pos_atom_4[:, 1] = pos_atom_1[:, 1] 
        pos_atom_4[:, 2] = pos_atom_1[:, 2] + length_uc/2.0

        self.positions = np.vstack((pos_atom_1, pos_atom_2, pos_atom_3, 
                                    pos_atom_4))

    def __init_velocities(self, sim):
        # Maxwell distributed velocity
        mu = 0
        sigma = np.sqrt(sim.space.temperature)
        vel = np.random.normal(mu, sigma, (self.num_particles, 3)) 

        # normalise the velocity so that the mean velocity is zero
        mean_velocity = np.mean(vel, axis = 0)
        vel -= mean_velocity 
        self.velocities = vel

    def calc_forces(self, sim):
        self.forces = calculate_lj(self.positions, sim.space.dimensions)

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

class Measurement(ABC):
    # Base class for a measurement
    def __init__(self):
        self.meas_data = None

    def initialize(self, sim):
        pass

    def print(self):
        pass

    def visualize(self):
        pass

    def measure(self, sim):
        pass

class KinEnergy_Meas(Measurement):
    def __init__(self):
        self.meas_data = None
    
    def print(self):
        pass

    def visualize(self):
        pass

    def measure(self, sim):
        pass

@jit
def calculate_lj(pos, dimens):
    # Pairwise interaction based on the Lennard-Jones potential
    # Cutoff 
    cutoff = 2.5
    # Calculate empty variables
    num_particles = pos.shape[0]
    LJ_force_x = np.zeros((num_particles, 1))
    LJ_force_y = np.zeros((num_particles, 1))
    LJ_force_z = np.zeros((num_particles, 1))
    LJ_pot = 0
    virial = 0

    # cutoff distance at 2.5*sigma
    inv_cutoff_sq = 1.0/(cutoff * cutoff) 
    # Loop over the number of atom pairs
    for ii in range(num_particles):
        for jj in range(ii+1, num_particles):

            # Distance between particles in the x, y, and z direction
            deltax = pos[ii, 0] - pos[jj, 0]
            deltay = pos[ii, 1] - pos[jj, 1]
            deltaz = pos[ii, 2] - pos[jj, 2]

            # Add/subtract L if image particle is closer than original. This is according to the repeated boundary condition
            deltax -= dimens[0] * np.rint(deltax / dimens[0])
            deltay -= dimens[1] * np.rint(deltay / dimens[1])
            deltaz -= dimens[2] * np.rint(deltaz / dimens[2])

            # 1 over the absolute value of distance between particles, squared.
            inv_dr_sq = 1.0/(deltax*deltax + deltay*deltay + deltaz*deltaz)

            # Potential and force at cutoff range 2.5*sigma
            if inv_dr_sq > inv_cutoff_sq:

                # Force vector between particle ii and jj
                Fx = (48*inv_dr_sq**7 - 24*inv_dr_sq**4) * deltax
                Fy = (48*inv_dr_sq**7 - 24*inv_dr_sq**4) * deltay
                Fz = (48*inv_dr_sq**7 - 24*inv_dr_sq**4) * deltaz

            else:
                # The distance is big enough for the force to be neglected
                Fx = 0
                Fy = 0
                Fz = 0

            # Sum the forces in the force vector for the corresponding particles
            LJ_force_x[ii] += Fx
            LJ_force_x[jj] -= Fx

            LJ_force_y[ii] += Fy
            LJ_force_y[jj] -= Fy

            LJ_force_z[ii] += Fz
            LJ_force_z[jj] -= Fz

    # stack x, y, and z components to create a force vector
    force = np.hstack((LJ_force_x, LJ_force_y, LJ_force_z)) 
    return force

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
    measurements = [KinEnergy_Meas()]

    sim = Simulation(dt=dt, num_steps=num_steps, particles=particles,
                     space=space, boundaries=boundaries,
                     measurements=measurements)
    sim.init_simulation()
    sim.run()


# class Config():
#     __instance = None
# 
#     def __init__(self):
#         if Config.__instance != None:
#             raise Exception("This class is a singleton!")
#         else:
#             self.parameters = {}
#             Config.__instance = self
# 
#     @staticmethod
#     def get_instance():
#         if Config.__instance == None:
#             Config()
#         return Config.__instance
# 
#     def set_parameter(self, param, value):
#         self.parameters[param] = value
# 
#     def get_parameter(self, param):
#         if self.parameters[param] != None:
#             return self.parameters[param]
#         else:
#             raise Exception("Parameter ", param, " is not defined")


