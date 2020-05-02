import numpy as np
from numba import jit

class Environment():
    def __init__(self, dimens, temp):
        self.dimens = np.array(dimens)
        self.temp = temp
        self.thermostat = None
        self.pressure = None

    def check_boundary(self, particles):
        particles.pos[:, 0] = particles.pos[:, 0] % self.dimens[0]
        particles.pos[:, 1] = particles.pos[:, 1] % self.dimens[1]
        particles.pos[:, 2] = particles.pos[:, 2] % self.dimens[2]

    def set_thermostat(self, thermostat_obj):
        self.thermostat = thermostat_obj

    def apply_regulators(self, particles):
        self.thermostat.apply(particles)

    def update_properties(self, particles):
        self.temp = 2/(3*particles.num_particles) * particles.kin_energy

class Thermostat():
    def __init__(self, temp, t_eq, t_interval):
        self.temp = temp
        self.t_eq = t_eq
        self.t_interval = t_interval

    def apply(self, particles):
        # Adjust velocity to acquire a constant temperature TDesired.
        # After tEquilibrium system should be in equilibrium with a somewhat constant T
        kin_energy = particles.kin_energy
        num_particles = particles.num_particles
        t = particles.steps_passed
        if t < self.t_eq and (t % self.t_interval == 0):
            particles.vel = particles.vel \
                    * np.sqrt((num_particles-1)*3*self.temp / (2*kin_energy))

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

            # Add/subtract L if image particle is closer than original. 
            # This is according to the repeated boundary condition
            deltax -= dimens[0] * np.rint(deltax / dimens[0])
            deltay -= dimens[1] * np.rint(deltay / dimens[1])
            deltaz -= dimens[2] * np.rint(deltaz / dimens[2])

            # 1 over the absolute value of distance between particles, squared.
            inv_dr_sq = 1.0/(deltax*deltax + deltay*deltay + deltaz*deltaz)

            # Potential and force at cutoff range 2.5*sigma
            if inv_dr_sq > inv_cutoff_sq:
                # Lennard-Jones potential energy within the volume
                LJ_pot += 4*(inv_dr_sq**6 - inv_dr_sq**3) 

                # Force vector between particle ii and jj
                Fx = (48*inv_dr_sq**7 - 24*inv_dr_sq**4) * deltax
                Fy = (48*inv_dr_sq**7 - 24*inv_dr_sq**4) * deltay
                Fz = (48*inv_dr_sq**7 - 24*inv_dr_sq**4) * deltaz

                # Sum the inner product of (dr . Force), i.e. the virial
                virial += -48*inv_dr_sq**6 + 24*inv_dr_sq**3 

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

            #  Create histogram of the number of particles at a seperation distance
            # histogram[int(1 / (np.sqrt(inv_dr_sq) * binLength))] += 1

    # stack x, y, and z components to create a force vector
    force = np.hstack((LJ_force_x, LJ_force_y, LJ_force_z)) 
    pot_energy = LJ_pot
    return force, pot_energy, virial

class ParticlesLJ():
    def __init__(self, num_particles, environment):
        self.num_particles = num_particles
        self.envir = environment
        self.steps_passed = 0
        self.time_passed = 0
        
        self.pos = self.__init_positions() 
        self.vel = self.__init_velocities()

        self.calculate_force()

        self.kin_energy = 0.5 * np.sum(self.vel ** 2)
        # self.pot_energy is calculated in self.calculate_force()

    def __init_positions(self):
        # The volume is assumed to be cubic; only the first argument of
        # self.envir.dimens is used.
        self.envir.dimens = [self.envir.dimens[0]] * len(self.envir.dimens)
        self.envir.dimens = np.array(self.envir.dimens)

        # the initial position is packed fcc.
        # 4 particles per fcc unit cell; number of unit cells in each dimension
        # should be equal.
        num_uc_onedim = round((self.num_particles//4)**(1.0/3.0))
        num_uc = num_uc_onedim ** 3
        self.num_particles = num_uc * 4
        print('the number of particles in the simulation is:', self.num_particles)

        length_uc = self.envir.dimens[0] / num_uc_onedim

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

        pos = np.vstack((pos_atom_1, pos_atom_2, pos_atom_3, pos_atom_4))
        return pos

    def __init_velocities(self):
        # determine the initial velocity of the particles.
        # every component of the velocity is drawn from the normal distribution,
        # such that the magnitude of the velocity is maxwell-boltzmann distributed.
        mu = 0
        sigma = np.sqrt(self.envir.temp)
        vel = np.random.normal(mu, sigma, (self.num_particles, 3)) 

        # normalise the velocity so that the mean velocity is zero
        mean_velocity = np.mean(vel, axis = 0)
        vel -= mean_velocity 
        return vel

    def calculate_force(self):
        self.force, self.pot_energy, self.virial = \
                calculate_lj(self.pos, self.envir.dimens) 

    def take_step(self, dt):
        # Move the particle according to the Verlet algorithm
        self.vel += 0.5 * self.force * dt
        self.pos += self.vel * dt
        self.envir.check_boundary(self)

        self.calculate_force()
        self.vel += 0.5 * self.force * dt
        self.kin_energy = 0.5 * np.sum(self.vel * self.vel) # Kinetc energy for this timestep

        self.envir.update_properties(self)
        self.envir.apply_regulators(self)

        self.steps_passed += 1
        self.time_passed += dt

