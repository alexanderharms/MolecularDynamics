# Originally made in collaboration with Aloys Erkelens as an assigment
# for the course Computational Physics by J.M. Thijssen in 2015/2016.

import numpy as np
from numba import jit

def initialize(Nuc, rho, T, Nbins):
    # Initializes the system to return the initial position and velocity of the particles

    N = 4*Nuc**3 # Total number of particles within the volume. 4 particles per FCC unit cell
    print('The number of particles is:', N)

    V = N/rho # Volume
    print('The volume is:', round(V, 4))

    L = V**(1/3) # Length of the volume in one dimension
    print('The length of the cubic volume is:', round(L, 4))

    Luc = L/Nuc # Length of a unit cell in one dimension

    binLength = L / Nbins # Binlength for the histogram of correlation function g(r)

    initPosition = Initial_Position(L, Luc, Nuc) # Calculate the initial positions
    initVelocity = Initial_Velocity(T, N) # Calculate the initial velocities

    return initPosition, initVelocity, V, N, L, Luc, binLength


def Initial_Position(L, Luc, Nuc):
    # Determines the initial positions for the Argon atoms in a FCC structure.

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


def Initial_Velocity(T, N):
    # Determine the initial velocity of the particles.
    # Every component of the velocity is drawn from the normal distribution,
    # such that the magnitude of the velocity is Maxwell-Boltzmann distributed.

    mu = 0
    sigma = np.sqrt(T)
    Velocity = np.random.normal(mu, sigma, (N, 3)) # Maxwell distributed velocity

    meanVelocity = np.mean(Velocity, axis = 0)

    Velocity -= meanVelocity # Normalise the velocity so that the mean velocity is zero

    return Velocity

@jit # Compiles the following function to machine code for enhanced computation speed
def DistanceForce(pos, N, L, Nbins, binLength):
    # The function DistanceForce calculates the distance between the particles
    # and then calculates the force between the two particles.
    # Within the for-loop the calculations are done component-wise to avoid the use of arrays.
    # This speeds up the computation time.
    # A histogram is created for number of particles at a seperation distances between the particles.

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
            histogram[int(1 / (np.sqrt(dr2) * binLength))] += 1

    LJForce = np.hstack((LJForcex, LJForcey, LJForcez)) # stack x, y, and z components to create a force vector

    return LJForce, LJPotential, drF, histogram

def Heat_Capacity(kinEnergy, N, TimeBlock, NumberOfBlocks):
    # Calculates the specific heat capacity from the variance of the kinetic energy with the Lebowitz formula.

    # Create empty arrays
    kinBlock = np.zeros(TimeBlock)
    Cv = np.zeros(NumberOfBlocks)
    CvPre = np.zeros(NumberOfBlocks)

    # Measure the Cv for each TimeBlock
    for ii in range(NumberOfBlocks):
        kinBlock = kinEnergy[ii*TimeBlock : (ii+1)*TimeBlock] # Kinetic energy within TimeBlock

        # Lebowitz formula. Cv in units of kB. Divide by volume to get Cv = (3/2)*(N/V) [kB]
        Cv[ii] = 1/((2/(3*N)) - (np.var(kinBlock) / (np.mean(kinBlock)**2)))
        CvPre[ii] = Cv[ii] / N # Prefactor Cv/N: should be 3/2 for a gas and 3 for a solid.

    MeanCv = np.mean(Cv) # Calculate the mean from all time blocks
    MeanCvPre = MeanCv/N

    ErrorCv = np.std(Cv)/np.sqrt(NumberOfBlocks) # Calculate the error
    ErrorCvPre = ErrorCv/N

    return MeanCv, MeanCvPre, ErrorCv, ErrorCvPre

def histogram_g(histogram, Nbins, binLength, V, N, tEquilibrium, TimeBlock, NumberOfBlocks):

    # Create empty arrays
    g = np.zeros((NumberOfBlocks, Nbins))
    r = np.zeros(Nbins)

    histogram = np.delete(histogram, np.s_[0:tEquilibrium], axis = 0) # Delete timesteps before equilibrium

    # Create the histogram for each TimeBlock
    for ii in range(NumberOfBlocks):
        # Average the histogram within each time block
        timeavgHistogram = np.mean(histogram[ii*TimeBlock : (ii+1)*TimeBlock, :], axis = 0)

        for jj in range(Nbins):
            r[jj] = binLength * (jj + 0.5) # Calculate distance vector
            g[ii, jj] = 2*V/(N*(N-1)) * timeavgHistogram[jj] / (4*np.pi*r[jj]**2*binLength) # Correlation function g(r)

    meang = np.mean(g, axis = 0) # Calculate the mean from all time blocks
    errorg = np.std(g, axis = 0) / np.sqrt(NumberOfBlocks) # Calculate the error

    return meang, errorg, r

def Calculate_Pressure(drF, actualT, N, L, TimeBlock, NumberOfBlocks):

    Pressure = np.zeros(NumberOfBlocks) # Create empty array

    # Measure the pressure for each TimeBlock
    for ii in range(NumberOfBlocks):
        # Pressure calculation, corrected for the cutoff potential
        Pressure[ii] = np.average(1.0 - drF[ii*TimeBlock : (ii+1)*TimeBlock] / (3*N*actualT[ii*TimeBlock : (ii+1)*TimeBlock])
                               - 2*np.pi*N/(3*actualT[ii*TimeBlock : (ii+1)*TimeBlock]*L**3) * 0.5106)

    meanPressure = np.mean(Pressure) # Calculate the mean from all timeblocks

    errorPressure = np.std(Pressure) / np.sqrt(NumberOfBlocks) # Calculate the error

    return meanPressure, errorPressure

def simulate(V, N, L, T, tEquilibrium, pos, vel, n_t, dt, Nbins, binLength, TimeBlock):
    # Runs the simulation by propagating the particles.
    # Extracts parameters from the simulation and prints the results.
    # Function returns all valuable variables of the simulation

    Force, Potential, drF, n = DistanceForce(pos, N, L, Nbins, binLength) # Force Potential and dr.F from initial positions

    # Create empty arrays
    kinEnergy = np.zeros(n_t)
    potEnergy = np.zeros(n_t)
    Pressure = np.zeros(n_t)
    drF = np.zeros(n_t)
    histogram = np.zeros((n_t, Nbins))

    print('\nsimulation is running\n')

    # Discrete time evolution of the particles
    for ii in range(n_t):
        # Move the particle according to the Verlet algorithm
        vel += 0.5 * Force * dt
        pos += vel * dt
        pos = pos % L # apply repeated boundary conditions

        # Calculate the new force of the particles, potential energy, the virial 'drF', and histogram.
        Force, potEnergy[ii], drF[ii], histogram[ii,:] = DistanceForce(pos, N, L, Nbins, binLength)

        vel += 0.5 * Force * dt

        kinEnergy[ii] = 0.5 * np.sum(vel * vel) # Kinetc energy for this timestep

        # Adjust velocity to acquire a constant temperature TDesired.
        # After tEquilibrium system should be in equilibrium with a somewhat constant T
        if ii < tEquilibrium and (ii % 10 == 0):
            vel = vel * np.sqrt((N-1)*3*T / (2*kinEnergy[ii]))

    actualT = 2/(3*N)*kinEnergy # Actual temperature during the simulation


    # Calculate several parameters from the simulation
    # The total measured time 'n_t' is devided by 'TimeBlock' into 'NumberOfBlocks' individual measurements
    NumberOfBlocks = int((n_t - tEquilibrium)/TimeBlock) # Calculate the number of measurement intervals in total time

    # Calculate the heat capacity Cv and its prefactor Cv/N with the corresponding errors
    Cv, CvPre, errorCv, errorCvPre = Heat_Capacity(kinEnergy[tEquilibrium:], N, TimeBlock, NumberOfBlocks)

    # Calculate the correlation function g(r) and the error
    g, errorg, r = histogram_g(histogram, Nbins, binLength, V, N, tEquilibrium, TimeBlock, NumberOfBlocks)

    # Calculate the pressure in the volume.
    Pressure, errorPressure = Calculate_Pressure(drF, actualT, N, L, TimeBlock, NumberOfBlocks)

    print('The number of measurements is: ', NumberOfBlocks, '\n')

    print('Heat Capacity: ',  round(Cv, 4), '+-',  round(errorCv, 4))
    print('Prefactor: ',  round(CvPre, 4), '+-', round(errorCvPre, 4))
    print('Pressure: ',  round(Pressure, 4), '+-', round(errorPressure, 4))
    print('Average Temperature: ',  round( np.average(actualT[tEquilibrium:]), 4))

    return pos, vel, potEnergy, kinEnergy, g, errorg, r, Cv, CvPre, errorCv, errorCvPre, Pressure, errorPressure
