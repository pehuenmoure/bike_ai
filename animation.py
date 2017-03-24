import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import math
import random

"""
Built of polygon class it allows for rotation
"""

WIDGHT = 5.
class bike_sim (plt.Polygon):
	def __init__ (self, bike):
		self.bike = bike
		self.x = self.bike.xy_coord[0]
		self.y = self.bike.xy_coord[2]
		self.height = self.bike.height
		self.width = WIDGHT
		self.theta = self.bike.lean_ang
		plt.Polygon.__init__(self, self.get_coordinates(self.x, self.y, self.theta), alpha = 0.5)

	def get_coordinates(self, x, y, theta):
		x1 = x - (self.width / 2.)*np.cos(theta)
		x2 = x - (self.height * np.sin(theta)) - (self.width / 2.)*np.cos(theta)
		x3 = x - (self.height * np.sin(theta)) + (self.width / 2.)*np.cos(theta)
		x4 = x + (self.width / 2.) * np.cos(theta)

		y1 = y - (self.width / 2.) * np.sin(theta)
		y2 = y + (self.height * np.cos(theta)) - (self.width / 2.)*np.sin(theta)
		y3 = y + (self.height * np.cos(theta)) + (self.width / 2.) * np.sin(theta)
		y4 = y + (self.width / 2.) * np.sin(theta)
		return ([[x1,y1], [x2,y2], [x3, y3], [x4,y4]])

	def update_command(self):
		self.x = self.bike.xy_coord[0]
		self.y = self.bike.xy_coord[2]
		self.height = self.bike.height
		self.width = 1.
		self.theta = self.bike.lean_ang
		self.set_xy(self.get_coordinates(self.x, self.y, self.theta))





class Simulator(object):
	def __init__(self, bike):
		""" Nav initializer """
		self.fig = plt.figure()
		plt.axis('equal')
		plt.axis([-70, 70, -5, 80])
		ax = plt.gca()
		self.fig.patch.set_facecolor('black')
		ax.patch.set_facecolor('black')
		ax.xaxis.set_visible(False)
		self.bike_sim = bike_sim(bike)
		ax.add_patch(self.bike_sim)
		ax.plot([-50, 50], [0, 0], color = 'white', alpha = 0.2)

	def run(self):
		anim = animation.FuncAnimation(self.fig, self.update_bike,
                                frames=50,
                                interval=5)

		plt.show()


	def update_bike(self, i):
		random.seed()
		self.bike_sim.bike.command(random.gauss(0,1), random.gauss(0,1))
		self.bike_sim.update_command()
		return self.bike_sim
