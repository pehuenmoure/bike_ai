import numpy as np
import math
import random
import pygame
from variables import *
from intersect import *
from time import *
from scipy.spatial import distance


''' the agent moves with constant speed '''
class Agent(object):
    def __init__(self, w0 = -0.5, w1 = -0.5, w2 = 0.5, w3 = 0.5, w4 = 0.5):
        ''' constructor
            pos : cartesian coordinate of center position
            dir : norm vector from the center
            ang : the turning angle of the agent [0,1](North) is 0, positive if clockwise
            isAlive : true if the agent is alive
            fitness : the score of the agent, updates when it's dead
            gene : the sequence of the commands that dictate the movements of agents
                   [-1]left,[0]straight, [1]right
            isElite : true if the agent is the best agent in the simulation so far

        '''
        self.weights = np.array([w0, w1, w2, w3, w4])
        self.pos = np.array([50., HEIGHT-50.])
        self.ang = 0
        self.dir = np.array([0.,-1.])
        self.isAlive = True
        self.fitness = 0
        gene = []
        for i in range(GENE):
            r = random.random()
            if r < (1/3):
                gene.append(-1)
            elif r < (2/3):
                gene.append(0)
            else:
                gene.append(1)
        self.gene = gene
        self.isElite = False
        self.sensors = np.array(5*[(0.0,0.0)])
        self.sensors2 = np.array(5*[-1.])
        self.offsets = np.arange(-180, 1, 45)
        # self.sensors = updateSensorData()
        self.sensorLength = SENSOR_LENGTH
        self.time = time()

    def setWeights(self, w0, w1, w2, w3, w4):
        self.weights = np.array([w0, w1, w2, w3, w4])

    def updateSensorData(self, walls):
        if self.dir[0] > 0:
            cosang = np.dot(np.array([0.,-1.]), self.dir)
            sinang = np.linalg.norm(np.cross(np.array([0.,-1.]), self.dir))
            theta = np.arctan2(sinang, cosang)
        else:
            cosang = np.dot(np.array([0.,-1.]), self.dir)
            sinang = np.linalg.norm(np.cross(np.array([0.,-1.]), self.dir))
            theta = -np.arctan2(sinang, cosang)

        thetas = np.rad2deg(theta) + self.offsets
        thetas = np.deg2rad(thetas)
        length = self.sensorLength

        edges = [([0,0],[0,HEIGHT]), ([0,0],[WIDTH,0]), ([WIDTH,0],[WIDTH,HEIGHT]), ([0,HEIGHT],[WIDTH,HEIGHT])]

        for i in np.arange(thetas.size):
            dx = self.pos[0] + length * np.cos(thetas[i])
            dy = self.pos[1] + length * np.sin(thetas[i])

            done = False
            for w in walls:
                if done: break
                wLineSeg = [(w.pos, w.pos+[w.width, 0]), (w.pos,w.pos+[0, w.height]), \
                 (w.pos+[w.width, 0], w.pos+[w.width,w.height]), \
                 (w.pos+[0, w.height], w.pos+[w.width,w.height])]
                wLineSeg = wLineSeg + edges
                for ls in wLineSeg:
                    p = calculateIntersect( \
                        ((self.pos[0],self.pos[1]),(dx,dy)), \
                        ((ls[0][0],ls[0][1]),(ls[1][0], ls[1][1])) )
                    if p is None:
                        self.sensors[i] = (dx,dy)
                        self.sensors2[i] = -1.
                    else:
                        self.sensors[i] = p
                        self.sensors2[i] = distance.euclidean(p,(self.pos[0],self.pos[1]))
                        done = True
                        break


    def draw(self, screen):
        ''' draw current state of agent onto the screen '''
        x = self.pos[0]
        y = self.pos[1]
        ''' get the angle between np.array([0,1]) and direction '''
        if self.dir[0] > 0:
            cosang = np.dot(np.array([0.,-1.]), self.dir)
            sinang = np.linalg.norm(np.cross(np.array([0.,-1.]), self.dir))
            theta = np.arctan2(sinang, cosang)

            ny = -AGENT_H * math.cos(theta-np.radians(self.ang))
            nx = AGENT_H * math.sin(theta-np.radians(self.ang))
            dx = AGENT_W * math.cos(theta-np.radians(self.ang))
            dy = -AGENT_W * math.sin(theta-np.radians(self.ang))
        else:
            cosang = np.dot(np.array([0.,-1.]), self.dir)
            sinang = np.linalg.norm(np.cross(np.array([0.,-1.]), self.dir))
            theta = -np.arctan2(sinang, cosang)

            ny = -AGENT_H * math.cos(theta-np.radians(self.ang))
            nx = AGENT_H * math.sin(theta-np.radians(self.ang))
            dx = AGENT_W * math.cos(theta-np.radians(self.ang))
            dy = -AGENT_W * math.sin(theta-np.radians(self.ang))

        for i in range(5):
            color = COLOR_C if self.sensors2[i] == -1. else COLOR_S
            pygame.draw.line(screen, color, self.pos, self.sensors[i], 1)
        # print(self.sensors2)
        # sleep(0.05)

        if self.isElite:
            pygame.draw.polygon(screen,(125,0,125),[[x+nx,y+ny],[x-nx-dx,y-ny+dy],[x-nx+dx,y-ny-dy]])
        else:
            pygame.draw.polygon(screen, COLOR ,[[x+nx,y+ny],[x-nx-dx,y-ny+dy],[x-nx+dx,y-ny-dy]])
        pygame.draw.circle(screen,COLOR_C,[int(x),int(y)],2)

    def update(self, screen, command, draw, walls, step = 0):
        ''' update based on the command
            -1: left
            0 : forward
            1 : right
            move the agent to the direction based on command
        '''
        if (self.isAlive):
            # if command == -1:
            #     self.ang = THETA
            # elif command == 1:
            #     self.ang = -THETA
            # elif command == 0:
            #     self.ang = 0
            self.ang = command

            theta = np.radians(self.ang)
            c, s = np.cos(theta), np.sin(theta)
            R = np.matrix([[c, -s], [s, c]])
            self.dir = np.squeeze(np.asarray(self.dir * R))
            self.pos += (self.dir)*VEL
            # kill the agent if it goes outside of the window
            if self.pos[0] < WIDTH and self.pos[0] > 0 and self.pos[1] > 0 and self.pos[1] < HEIGHT:
                self.isAlive = True
            else:
                self.getFitness(step)
                self.isAlive = False

            self.updateSensorData(walls)
        if (draw):
            self.draw(screen)

    def getCommand(self, idx = 0):
        ''' read the currect position of gene '''
        # if len(self.gene) > idx:
        #     return self.gene[idx]
        # else:
        #     return None
        return self.weights.dot(self.sensors2.T)


    def dead(self):
        ''' kill the agent '''
        self.isAlive = False

    def getFitness(self, step = 0):
        ''' evaluation the score of the agent '''
        # dist = np.linalg.norm(self.pos - np.array([780, 20])) # distance from the target
        # if dist > 0:
        #     self.fitness = dist
        # # after reach the target, try to minimize the step
        # else:
        #     self.fitness = dist - (1000-step)
        self.fitness = time() - self.time

    def setGene(self, gene):
        ''' set the gene of the agent '''
        self.gene = gene

    def setElite(self):
        ''' set the elite status of the agent '''
        self.isElite = True


class Target(object):
    def __init__(self, pos = np.array([780, 20])):
        self.pos = pos

    def draw(self, screen):
        pygame.draw.rect(screen,(0,0,128),[self.pos[0]- 10, self.pos[1] - 10, 20, 20])


class Wall(object):
    def __init__(self, move = False):
        position = np.array([random.random() * WIDTH, 50 + random.random() * (HEIGHT -100)])
        self.init = np.copy(position)
        self.pos = position
        self.width = 100 + 100 * random.random()
        self.height = 10
        self.vel = 0
        if (move):
            random.seed()
            self.vel = random.uniform(-1,1)
        # print(self.pos, self.width, self.height)

    def update(self, screen, draw):
        self.pos[0] += self.vel
        if self.pos[0] < 0:
            self.pos[0] = 0
            self.vel = -1 * self.vel
        elif self.pos[0]+self.width > WIDTH:
            self.pos[0] = WIDTH-self.width
            self.vel = -1 * self.vel
        if draw:
            self.draw(screen)

    def draw(self, screen):
        pygame.draw.rect(screen,(128,128,128),[self.pos[0], self.pos[1], self.width, self.height])

    def reset(self):
        self.pos = self.init
