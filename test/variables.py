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
THETA = 3 # turning angle
GENE = 450 # the length of chromosome (sequence of commands)
HEALTH = 1000 # the fuel of the agent
NUTRIENT = 0 # the fuel recovers by picking up food

MAXSTEP = 3000 # give runner the chance to win

# Simulation
MUTATION = 0.1 # mutation rate
WALL = 15 # number of walls
WALL_MOVE = False # stationary walls
FREQ = 10 # animation frequency (every 100 generation)

FOOD = 0 # number of food generated in the map

# Sensor
RADIUS = 50

# Network Parameters
n_hidden_1 =  3 # 1st layer number of features
n_hidden_2 =  3 # 2nd layer number of features
n_input =  7    # input
n_classes = 3   # left, straight, right

COLOR_C = (0,0,255) # the center circle of the agents
COLOR_S = (255,0,0) # the color of the sensor
