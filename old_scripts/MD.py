# Originally made in collaboration with Aloys Erkelens as an assigment
# for the course Computational Physics by J.M. Thijssen in 2015/2016.

import numpy as np
from numba import jit
import matplotlib.pyplot as plt

class MolecularDynamics():

    def __init__(self, dt, n_t, t_equilibrium, Nuc, Nbins, time_block, rho, T):
        self.dt = dt
        self.n_t = n_t
        self.t_equilibrium = t_equilibrium

        self.Nbins = Nbins
        self.time_block = time_block

        self.N = 4*Nuc**3 # Total number of particles within the volume. 4 particles per FCC unit cell
        print('The number of particles is:', self.N)
        self.Nuc = Nuc

        self.rho = rho
        self.T = T

        self.initialize()

    def initialize(self):
        """Initializes the system to return the initial position and velocity of the particles"""

        print('The number of particles is:', self.N)

        self.V = self.N/ self.rho # Volume
        print('The volume is:', round(self.V, 4))

        self.L = self.V**(1.0/3.0) # Length of the volume in one dimension
        print('The length of the cubic volume is:', round(self.L, 4))

        self.Luc = self.L / self.Nuc # Length of a unit cell in one dimension

        self.binlength = self.L / self.Nbins # Binlength for the histogram of correlation function g(r)

        self.pos = self.initial_position(self.L, self.Luc, self.Nuc) # Calculate the initial positions
        self.vel = self.initial_velocity(self.T, self.N) # Calculate the initial velocities

    def initial_position(self):
        """Determines the initial positions for the Argon atoms in a FCC structure."""

        L = self.L
        Luc = self.Luc
        Nuc = self.Nuc

        x = np.linspace(0, L - Luc, Nuc) # Position vector to fill the space
        xx, yy, zz = np.meshgrid(x, x, x) # Square meshgrid of positions for the atoms
        Nuc3 = Nuc**3 # Total number of unit cells in volume

        # Reshape meshgrid into a 1D arrays
        xPos = np.reshape(xx, (Nuc3, 1))
        yPos = np.reshape(yy, (Nuc3, 1))
        zPos = np.reshape(zz, (Nuc3, 1))

        # Shape position vectors for the first set of particles on the square grid
        pos1 = np.hstack((xPos, yPos, zPos))

        # Second set of position vectors. pos1 shifted in the x, y direction
        pos2 = np.zeros((Nuc3, 3))
        pos2[:, 0] = pos1[:, 0] + Luc/2
        pos2[:, 1] = pos1[:, 1] + Luc/2
        pos2[:, 2] = pos1[:, 2]

        # Third set of position vectors. pos1 shifted in the y, z direction
        pos3 = np.zeros((Nuc3, 3))
        pos3[:, 0] = pos1[:, 0]
        pos3[:, 1] = pos1[:, 1] + Luc/2
        pos3[:, 2] = pos1[:, 2] + Luc/2

        # Fourth set of position vectors. pos1 shifted in the x, z direction
        pos4 = np.zeros((Nuc3, 3))
        pos4[:, 0] = pos1[:, 0] + Luc/2
        pos4[:, 1] = pos1[:, 1]
        pos4[:, 2] = pos1[:, 2] + Luc/2

        pos = np.vstack((pos1, pos2, pos3, pos4)) # Position vector of all particles

        return pos

    def initial_velocity(self):
        """Determines the initial velocity of the particles.
        Every component of the velocity is drawn from the normal distribution,
        such that the magnitude of the velocity is Maxwell-Boltzmann distributed."""

        mu = 0
        sigma = np.sqrt(self.T)
        velocity = np.random.normal(mu, sigma, (self.N, 3)) # Maxwell distributed velocity

        mean_velocity = np.mean(velocity, axis = 0)

        velocity -= mean_velocity # Normalise the velocity so that the mean velocity is zero

        return velocity

    def simulate(self):
        """Runs the simulation by propagating the particles.
        Extracts parameters from the simulation and prints the results.
        Function returns all valuable variables of the simulation"""
        V = self.V
        N = self.N
        L = self.L
        T = self.T
        t_equilibrium = self.t_equilibrium
        pos = self.pos
        vel = self.vel
        n_t = self.n_t
        dt = self.dt
        Nbins = self.Nbins
        binlength = self.binlength
        time_block = self.time_block

        force, potential, drF, n = self.distanceforce(pos) # Force, potential and dr.F from initial positions

        # Create empty arrays
        kin_energy = np.zeros(n_t)
        pot_energy = np.zeros(n_t)
        pressure = np.zeros(n_t)
        drF = np.zeros(n_t)
        histogram = np.zeros((n_t, Nbins))

        print('\nsimulation is running\n')

        # Discrete time evolution of the particles
        for ii in range(n_t):
            # Move the particle according to the Verlet algorithm
            vel += 0.5 * force * dt
            pos += vel * dt
            pos = pos % L # apply repeated boundary conditions

            # Calculate the new force of the particles, potential energy, the virial 'drF', and histogram.
            force, pot_energy[ii], drF[ii], histogram[ii,:] = self.distanceforce(pos)

            vel += 0.5 * force * dt

            kin_energy[ii] = 0.5 * np.sum(vel * vel) # Kinetc energy for this timestep

            # Adjust velocity to acquire a constant temperature TDesired.
            # After tEquilibrium system should be in equilibrium with a somewhat constant T
            if ii < t_equilibrium and (ii % 10 == 0):
                vel = vel * np.sqrt((N-1)*3*T / (2*kin_energy[ii]))
        self.kin_energy = kin_energy
        self.pot_energy = pot_energy

        self.processing(drF, historgram)

    def processing(self, drF, histogram):
        """ Processes the results of the simulations into Cv, g(r) and pressure """
        n_t = self.n_t
        t_equilibrium = self.t_equilibrium
        time_block = self.time_block

        self.T_actual = 2/(3*self.N) * self.kin_energy # Actual temperature during the simulation

        # Calculate several parameters from the simulation
        # The total measured time 'n_t' is divided by 'time_block' into 'number_blocks' individual measurements
        self.number_blocks = int((n_t - t_equilibrium)/time_block) # Calculate the number of measurement intervals in total time

        # Calculate the heat capacity Cv and its prefactor Cv/N with the corresponding errors
        self.heat_capacity(self.kin_energy[t_equilibrium:])

        # Calculate the correlation function g(r) and the error
        self.histogram_g(histogram)

        # Calculate the pressure in the volume.
        self.calculate_pressure(drF)

    @jit # Compiles the following function to machine code for enhanced computation speed
    def distanceforce(self, pos):
        """ The function DistanceForce calculates the distance between the particles
        and then calculates the force between the two particles.
        Within the for-loop the calculations are done component-wise to avoid the use of arrays.
        This speeds up the computation time.
        A histogram is created for number of particles at a seperation distances between the particles."""

        N = self.N
        L = self.L
        Nbins = self.Nbins
        binlength = self.binlength

        # Create empty arrays and variables
        LJForcex = np.zeros((N, 1))
        LJForcey = np.zeros((N, 1))
        LJForcez = np.zeros((N, 1))
        histogram = np.zeros(Nbins)
        LJPotential = 0
        drF = 0

        rCutoff2 = 1/(2.5*2.5) # cutoff distance at 2.5*sigma

        # Loop over the number of atom pairs
        for ii in range(N):
            for jj in range(ii+1, N):

                # Distance between particles in the x, y, and z direction
                deltax = pos[ii, 0] - pos[jj, 0]
                deltay = pos[ii, 1] - pos[jj, 1]
                deltaz = pos[ii, 2] - pos[jj, 2]

                # Add/subtract L if image particle is closer than original. This is according to the repeated boundary condition
                deltax -= L * np.rint(deltax / L)
                deltay -= L * np.rint(deltay / L)
                deltaz -= L * np.rint(deltaz / L)

                # 1 over the absolute value of distance between particles, squared.
                dr2 = 1/(deltax*deltax + deltay*deltay + deltaz*deltaz)

                # Potential and force at cutoff range 2.5*sigma
                if dr2 > rCutoff2:

                    LJPotential += 4*(dr2**6 - dr2**3) # Lennard-Jones potential energy within the volume

                    # Force vector between particle ii and jj
                    Fx = (48*dr2**7 - 24*dr2**4) * deltax
                    Fy = (48*dr2**7 - 24*dr2**4) * deltay
                    Fz = (48*dr2**7 - 24*dr2**4) * deltaz

                    drF += -48*dr2**6 + 24*dr2**3 # Sum the inner product of (dr . Force), i.e. the virial

                else:
                    # The distance is big enough for the force to be neglected
                    Fx = 0
                    Fy = 0
                    Fz = 0

                # Sum the forces in the force vector for the corresponding particles
                LJForcex[ii] += Fx
                LJForcex[jj] -= Fx

                LJForcey[ii] += Fy
                LJForcey[jj] -= Fy

                LJForcez[ii] += Fz
                LJForcez[jj] -= Fz

                #  Create histogram of the number of particles at a seperation distance
                histogram[int(1 / (np.sqrt(dr2) * binlength))] += 1

        LJForce = np.hstack((LJForcex, LJForcey, LJForcez)) # stack x, y, and z components to create a force vector

        return LJForce, LJPotential, drF, histogram

    def heat_capacity(self, kin_energy):
        """Calculates the specific heat capacity from the variance of the kinetic
        energy with the Lebowitz formula."""
        N = self.N
        time_block = self.time_block
        number_blocks = self.number_blocks

        # Create empty arrays
        kin_block = np.zeros(time_block)
        Cv = np.zeros(number_blocks)
        CvPre = np.zeros(number_blocks)

        # Measure the Cv for each time block
        for ii in range(number_blocks):
            kin_block = kin_energy[ii*time_block : (ii+1)*time_block] # Kinetic energy within time_block

            # Lebowitz formula. Cv in units of kB. Divide by volume to get Cv = (3/2)*(N/V) [kB]
            Cv[ii] = 1/((2/(3*N)) - (np.var(kin_block) / (np.mean(kin_block)**2)))
            CvPre[ii] = Cv[ii] / N # Prefactor Cv/N: should be 3/2 for a gas and 3 for a solid.

        self.MeanCv = np.mean(Cv) # Calculate the mean from all time blocks
        self.MeanCvPre = MeanCv/N

        self.ErrorCv = np.std(Cv)/np.sqrt(number_blocks) # Calculate the error
        self.ErrorCvPre = ErrorCv/N

    def histogram_g(self, histogram):
        """ Calculates the g factor """
        Nbins = self.Nbins
        binlength = self.binlength
        V = self.V
        N = self.N
        t_equilibrium = self.t_equilibrium
        time_block = self.time_block
        number_blocks = self.number_blocks

        # Create empty arrays
        g = np.zeros((number_blocks, Nbins))
        r = np.zeros(Nbins)

        histogram = np.delete(histogram, np.s_[0:t_equilibrium], axis = 0) # Delete timesteps before equilibrium

        # Create the histogram for each time block
        for ii in range(number_blocks):
            # Average the histogram within each time block
            timeavgHistogram = np.mean(histogram[ii*time_block : (ii+1)*time_block, :], axis = 0)

            for jj in range(Nbins):
                r[jj] = binlength * (jj + 0.5) # Calculate distance vector
                g[ii, jj] = 2*V/(N*(N-1)) * timeavgHistogram[jj] / (4*np.pi*r[jj]**2*binlength) # Correlation function g(r)

        self.meang = np.mean(g, axis = 0) # Calculate the mean from all time blocks
        self.errorg = np.std(g, axis = 0) / np.sqrt(number_blocks) # Calculate the error
        self.r = r

    def calculate_pressure(self, drF):
        """ Calculates pressure for NVT ensemble """
        N = self.N
        L = self.L
        time_block = self.time_block
        number_blocks = self.number_blocks
        T_actual = self.T_actual
        pressure = np.zeros(number_blocks) # Create empty array

        # Measure the pressure for each time block
        for ii in range(number_blocks):
            # Pressure calculation, corrected for the cutoff potential
            pressure[ii] = np.average(1.0 - drF[ii*time_block : (ii+1)*time_block]
                                    / (3*N*T_actual[ii*time_block : (ii+1)*time_block])
                                    - 2*np.pi*N/(3*T_actual[ii*time_block : (ii+1)*time_block]*L**3) * 0.5106)

        self.meanPressure = np.mean(pressure) # Calculate the mean from all timeblocks
        self.errorPressure = np.std(pressure) / np.sqrt(number_blocks) # Calculate the error
