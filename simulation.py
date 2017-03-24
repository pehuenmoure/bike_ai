from animation import *
from agent import *

if __name__ == '__main__':
    new_bike = Bike(lean_ang = 0.0)
    simulation = Simulator(new_bike)
    simulation.run()
