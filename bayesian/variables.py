''' Global variables '''

# the frame dimension
WIDTH = 800.
HEIGHT = 600.

# Agent
VEL = 3.0   # constant speed of the agents
AGENT_W = 10 # width of the agents
AGENT_H = 10 # height of the agents
COLOR = (0,255,0) # color of the agents
COLOR_C = (0,0,255) # the center circle of the agents
COLOR_S = (255,0,0) # the color of the sensor
THETA = 3 # turning angle
GENE = 400 # the length of chromosome (sequence of commands)
SENSOR_LENGTH = 50

# Simulation
MUTATION = 0.01 # mutation rate
WALL = 20 # number of walls
WALL_MOVE = True # stationary walls
FREQ = 5 # animation frequency (every 100 generation)
MAX_STEPS = 10000
