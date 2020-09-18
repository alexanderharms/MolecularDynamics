from abc import ABC
import numpy as np

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
        self.meas_data = [None] * (sim.num_steps // self.interval)
        
    def print(self):
        print("Kinetic energy: ", self.measurement)

    def visualize(self):
        pass

    def measure(self, sim):
        self.measurement = 0.5 * np.sum(sim.particles.velocities ** 2)
        self.meas_data[sim.time_steps // self.interval] = self.measurement
        if self.realtime_print:
            self.print()
