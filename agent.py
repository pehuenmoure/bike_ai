import numpy as np
import math
import random

# constants
G = 50.
MASS = 100.
HEIGHT = 40.
TIMESTEP = 0.05
LENGTH = 2.

MAX_VELOCITY = 100.
MAX_ACCELERATION = 3.

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

	def __init__(self, xy_coord = [0.,0.,0.], direction = [0.,1., 0.], \
				steering_vector = [0., 0., 0.], speed = 0., acc = 0.0, \
				lean_ang = 0., steer_ang = 0.):

		""" Bike initializer """
		self.xy_coord = xy_coord
		self.direction = direction
		self.steering_vector = steering_vector
		self.speed = speed
		self.acc = acc
		self.lean_ang = lean_ang	# in radians
		self.steer_ang = steer_ang	# in radians
		self.height = HEIGHT
		self.length = LENGTH
		self.mass = MASS
		self.omega = 0.
		self.center_mass = [0.,0.,0.]
		self.is_dead = False

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
		# catch the boundary
		if self.acc > MAX_ACCELERATION:
			self.acc = MAX_ACCELERATION
		elif self.acc < -MAX_ACCELERATION:
			self.acc = -MAX_ACCELERATION

		self.steer_ang += np.radians(delta_steer)

		self.update()

	def update(self):
		if(not self.is_dead):
			self.speed += self.acc
			# catch the boundary
			if self.speed > MAX_VELOCITY:
				self.speed = MAX_VELOCITY
			elif self.speed < -MAX_VELOCITY:
				self.speed = -MAX_VELOCITY

			self.steering_vector += \
				np.array([np.sin(self.steer_ang), np.cos(self.steer_ang), 0.])


			# new direction is norm(direction + steering vector)
			self.direction = (self.direction + self.steering_vector) / \
			np.linalg.norm (self.direction + self.steering_vector)

			# - update lean_ang
			# law of cosines
			if (self.steer_ang == 0 or self.speed == 0):
				if (np.array_equal(self.xy_coord, self.center_mass)):
					self.center_mass[0] += random.gauss(0,1)/10

				delta = self.center_mass[0] - self.xy_coord[0]

				# angular velocity
				self.omega += np.sin(self.height / delta / 8.) / 10
				self.lean_ang += self.omega * TIMESTEP

			else:
				side = math.sqrt(2. * (self.length / 2.)**2. - 2 * (self.length / 2.)**2. * math.cos(self.steer_ang))
				turning_radius = side / (2. * math.sin(self.steer_ang))
				print(turning_radius)
				self.lean_ang += -np.arctan(self.speed**2 / (turning_radius*1000)) / 100
				print(self.lean_ang)
				# (if center of mass if right on the position, add very small noise)

			# catch the boundary
			if (self.lean_ang > np.pi / 2.):
				self.lean_ang = np.pi / 2.
				self.is_dead = True
			elif (self.lean_ang < - np.pi / 2.):
				self.lean_ang = - np.pi / 2.
				self.is_dead = True

			# - update xy_coord
			self.xy_coord += self.direction * self.speed * TIMESTEP

			# catch the boundary
			if (self.xy_coord[0]>50):
				self.xy_coord[0]=50
				self.is_dead = True
			elif (self.xy_coord[0]<-50):
				self.xy_coord[0]=-50
				self.is_dead = True


bike = Bike()
print bike.vector
