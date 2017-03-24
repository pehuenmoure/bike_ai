import numpy as np
import math

# constants
G = 5.
MASS = 10.
HEIGHT = 8.

# Bike object represents the agent that our AI will study and
class Bike(object):
	"""
	INSTANCE ATTRIBUTES:

	OPTIONAL:
	direction [float]: direction of bicycle relative to the coordinate(0,0)
						2-D vector with the magnitude of 1
	xy_coord  [tuple]: initial xyz coordinate of bicycle (z = 0)
	lean_ang  [float]: initial lean angle of bicycle
	steer_ang [float]: initial steering angle of bicycle
	speed 	  [float]: initial speed of bicycle
	acc		  [float]: initial acceleration of bicycle

	PARAMETERS:
	height	  [float]: height of bicycle
	mass	  [float]: mass of bicycle
	m_center  [tuple]: 3D coordinate of center of the mass
	turning_r [float]: turning radius of the bicycle
	"""

	def __init__(self, xy_coord = (0.,0.,0.), direction = [1.,1.], \
				steering_vector = [1., 1.], speed = 0., acc = 1, \
				lean_ang = 0., steer_ang = 0.):
		""" Bike initializer """
		self.xy_coord = xy_coord
		self.direction = direction
		self.steering_vector = steering_vector
		self.speed = speed
		self.acc = acc
		self.lean_ang = lean_ang
		self.steer_ang = steer_ang
		self.height = HEIGHT
		self.mass = MASS
		self.turning_r = 0.

	@property
	def vector(self):
		b = np.array([math.cos(self.direction[0]), math.sin(self.direction[1])])
		return b/np.linalg.norm(b)

	def command(self, delta_acc, delta_steer):
		'''
		delta_acc	[float]: Change in acceleration, domain [-5., 5.]
		delta_steer	[float]: Change in steering angle domain [-10., 10.]
		The agent (bicycle) is only allowed to change the increment of
		acceleration and steering angle
		'''
		self.acc += delta_acc
		self.steer_ang += delta_steer

	def update(self):
		self.speed += self.acc

		# TODO
		# - update steering vector
		# - update lean_ang
		#		(if center of mass if right on the position, add very small noise)
		# - update direction
		# - update xy_coord
		# - update turning_r



bike = Bike()
print bike.vector
