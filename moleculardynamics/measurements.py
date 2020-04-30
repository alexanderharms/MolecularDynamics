import numpy as np
from numba import jit 

def calc_heat_capacity(kin_energy, num_particles, block_size, num_blocks):
    # Calculates the specific heat capacity from the variance of the kinetic energy with the Lebowitz formula.

    # Create empty arrays
    kin_block = np.zeros(block_size)
    cv = np.zeros(num_blocks)
    cv_factor = np.zeros(num_blocks)

    # Measure the cv for each block in time 
    for ii in range(num_blocks):
        # Kinetic energy within TimeBlock
        kin_block = kin_energy[ii*block_size : (ii+1)*block_size] 

        # Lebowitz formula. Cv in units of kB. Divide by volume to get Cv = (3/2)*(N/V) [kB]
        cv[ii] = 1/((2/(3*num_particles)) - (np.var(kin_block) / (np.mean(kin_block)**2)))
        # Prefactor Cv/N: should be 3/2 for a gas and 3 for a solid.
        cv_factor[ii] = cv[ii] / num_particles 

    cv_mean = np.mean(cv) # Calculate the mean from all time blocks
    cv_factor_mean = np.mean(cv_factor)

    cv_error = np.std(cv) / np.sqrt(num_blocks) # Calculate the error
    cv_factor_error = np.std(cv_factor) / np.sqrt(num_blocks) 

    print('Heat Capacity: ',  round(cv_mean, 4), '+-',  round(cv_error, 4))
    print('Heat Cap. Prefactor: ',  round(cv_factor_mean, 4), '+-', 
          round(cv_factor_error, 4))
    return [cv_mean, cv_error] , [cv_factor_mean, cv_factor_error]


def calc_g(hist_array, num_particles, dimens, num_bins, block_size, num_blocks):
    # Create empty arrays
    g = np.zeros((num_blocks, num_bins))
    r = np.zeros(num_bins)

    # Calculate volume
    vol = dimens[0] * dimens[1] * dimens[2]

    bin_length = dimens[0] / num_bins

    # Create the histogram for each time block 
    for block in range(num_blocks):
        # Average the histogram within each time block
        hist_avg = np.mean(hist_array[block*block_size : (block+1)*block_size, :], axis = 0)

        for bin_idx in range(num_bins):
            r[bin_idx] = bin_length * (bin_idx + 0.5) # Calculate distance vector
            g[block, bin_idx] = 2*vol/(num_particles*(num_particles-1)) \
                    * hist_avg[bin_idx] / (4*np.pi*r[bin_idx]**2*bin_length) # Correlation function g(r)

    g_mean = np.mean(g, axis = 0) # Calculate the mean from all time blocks
    g_error = np.std(g, axis = 0) / np.sqrt(num_blocks) # Calculate the error

    return g_mean, g_error, r

@jit
def calc_histogram(pos, dimens, num_bins):
    # Loop over the number of atom pairs
    num_particles = pos.shape[0]
    bin_length = dimens[0] / num_bins
    histogram_vec = np.zeros(num_bins)
    for ii in range(num_particles):
        for jj in range(ii+1, num_particles):

            # Distance between particles in the x, y, and z direction
            delta_x = pos[ii, 0] - pos[jj, 0]
            delta_y = pos[ii, 1] - pos[jj, 1]
            delta_z = pos[ii, 2] - pos[jj, 2]

            # Add/subtract L if image particle is closer than original. This is according to the repeated boundary condition
            delta_x -= dimens[0] * np.rint(delta_x / dimens[0])
            delta_y -= dimens[1] * np.rint(delta_y / dimens[1])
            delta_z -= dimens[2] * np.rint(delta_z / dimens[2])

            # 1 over the absolute value of distance between particles, squared.
            inv_dr_sq = 1/(delta_x*delta_x + delta_y*delta_y + delta_z*delta_z)

            histogram_vec[int(1 / (np.sqrt(inv_dr_sq) * bin_length))] += 1

    return histogram_vec
