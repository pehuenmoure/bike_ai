import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import math

"""
Built of polygon class it allows for rotation
"""
class bike_sim (plt.Polygon):
	def __init__ (self, bike):
		self.bike = bike
		self.x = bike.xy_coord[0]
		self.y = bike.xy_coord[1]
		self.height = bike.height
		self.width = 1.
		self.theta = bike.lean_ang
		plt.Polygon.__init__(self, self.get_coordinates(self.x, self.y, self.theta), alpha = 0.5)

	def get_coordinates(self, x, y, theta):
		x1 = x - (self.width / 2.)*math.cos(theta)
		x2 = x - (self.height * math.sin(theta)) - (self.width / 2.)*math.cos(theta)
		x3 = x - (self.height * math.sin(theta)) + (self.width / 2.)*math.cos(theta)
		x4 = x + (self.width / 2.) * math.cos(theta)

		y1 = y - (self.width / 2.) * math.sin(theta)
		y2 = y + (self.height * math.cos(theta)) - (self.width / 2.)*math.sin(theta)
		y3 = y + (self.height * math.cos(theta)) + (self.width / 2.) * math.sin(theta)
		y4 = y + (self.width / 2.) * math.sin(theta)
		return ([[x1,y1], [x2,y2], [x3, y3], [x4,y4]])


class Simulator(object):
	def __init__(self, bike):
		""" Nav initializer """
		self.fig = plt.figure()
		plt.axis('equal')
		plt.axis([-30, 30, -5, 30])

		ax = plt.gca()

		self.fig.patch.set_facecolor('black')
		ax.patch.set_facecolor('black')
		ax.xaxis.set_visible(False)

		self.bike_sim = bike_sim(bike)
		ax.add_patch(self.bike_sim)
		ax.plot([-30, 30], [0, 0], color = 'white', alpha = 0.2)

	def run(self):
		anim = animation.FuncAnimation(self.fig, self.update_bike,
                                frames=5000,
                                interval=5)
		plt.show()


	def update_bike(self, i):
		return self.bike_sim
