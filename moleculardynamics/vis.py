import numpy as np
import matplotlib.pyplot as plt

def plot_energy(num_steps, dt, kin_energy, pot_energy):
    # Plot the energy of the system
    t = np.linspace(1, num_steps, num_steps) * dt

    font = {'family' : 'serif', 'size'   : 18}
    plt.rc('font', **font)

    fig = plt.figure()
    plt.plot(t, pot_energy, label = 'Potential energy', linewidth = 2)
    plt.plot(t, kin_energy, label = 'Kinetic energy', linewidth = 2)
    plt.plot(t, kin_energy + pot_energy, label = 'Total energy', linewidth = 2)
    plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
    plt.ylabel('Energy')
    plt.xlabel('time')
    plt.title('Energy of the system')
    plt.tick_params(axis = 'both', pad=10)
    fig.savefig('./output/energy_rewrite.pdf', bbox_inches='tight')
    
def plot_g(r, g, g_error, length):
    fig = plt.figure()
    plt.plot(r, g, linewidth = 2)
    plt.axhline(y = 1, color = 'k', linestyle = 'dashed', linewidth = 2)
    plt.fill_between(r, g-g_error, g+g_error, alpha = 0.5, \
            edgecolor = '#CC4F1B', facecolor = '#FF9848')
    plt.ylabel('g(r)')
    plt.xlabel('r/$\sigma$')
    plt.title('Correlation function')
    plt.xlim(0, 0.5*length)
    plt.tick_params(axis = 'both', pad = 10)

    fig.savefig('./output/correlation.pdf', bbox_inches='tight')
