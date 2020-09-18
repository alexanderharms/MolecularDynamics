from abc import ABC
import numpy as np
import matplotlib.pyplot as plt

class Measurement(ABC):
    # Base class for a measurement
    def __init__(self, interval, realtime_print):
        self.measurement = None
        self.meas_data = None
        self.realtime_print = realtime_print 
        self.interval = interval

    def initialize(self, sim):
        pass

    def print(self):
        pass

    def visualize(self):
        pass

    def measure(self, sim):
        pass

class KinEnergy_Meas(Measurement):
    def __init__(self, interval=1, realtime_print=False):
        super(KinEnergy_Meas, self).__init__(interval, realtime_print)
    
    def initialize(self, sim):
        self.meas_data = np.zeros((sim.num_steps // self.interval, 2))
        
    def print(self):
        print("Time: ", self.measurement[0])
        print("Kinetic energy: ", self.measurement[1])

    def visualize(self):
        plt.figure()
        plt.plot(self.meas_data[:, 0], self.meas_data[:, 1])
        plt.show()

    def measure(self, sim):
        self.measurement = [sim.time, 
                            0.5 * np.sum(sim.particles.velocities**2)]
        self.meas_data[sim.time_steps // self.interval, :] = self.measurement
        if self.realtime_print:
            self.print()

class TotEnergy_Meas(Measurement):
    def __init__(self, interval=1, realtime_print=False):
        super(TotEnergy_Meas, self).__init__(interval, realtime_print)
    
    def initialize(self, sim):
        self.meas_data = np.zeros((sim.num_steps // self.interval, 4))
        
    def print(self):
        print("Time: ", self.measurement[1])
        print("Kinetic energy: ", self.measurement[1])
        print("Potential energy: ", self.measurement[2])
        print("Total energy: ", self.measurement[3])

    def visualize(self):
        plt.figure()
        plt.plot(self.meas_data[:, 0], self.meas_data[:, 1])
        plt.plot(self.meas_data[:, 0], self.meas_data[:, 2])
        plt.plot(self.meas_data[:, 0], self.meas_data[:, 3])
        plt.show()

    def measure(self, sim):
        self.measurement = [sim.time, 
                            0.5 * np.sum(sim.particles.velocities**2),
                            sim.particles.pot_energy, 0]
        self.measurement[3] = self.measurement[1] + self.measurement[2]
        self.meas_data[sim.time_steps // self.interval, :] = self.measurement
        if self.realtime_print:
            self.print()
