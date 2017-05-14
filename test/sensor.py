import numpy as np
import math
from variables import *
from agent import *
import multiprocessing as mp
import time

def pointInLineSeg (ls1, ls2, p):
    (X1, Y1), (X2, Y2) = ls1
    (X3, Y3), (X4, Y4) = ls2
    xVal, yVal = p
    xi1 = [min(X1,X2), max(X1,X2)]
    xi2 = [min(X3,X4), max(X3,X4)]
    yi1 = [min(Y1,Y2), max(Y1,Y2)]
    yi2 = [min(Y3,Y4), max(Y3,Y4)]
    xbool1 = min(xi1[1], xi2[1]) >= xVal
    xbool2 = max(xi1[0], xi2[0]) <= xVal
    ybool1 = min(yi1[1], yi2[1]) >= yVal
    ybool2 = max(yi1[0], yi2[0]) <= yVal
    if xbool1 and xbool2 and ybool1 and ybool2:
        return True
    else:
        return False

def slope (p1, p2):
    if p1[0] != p2[0]:
        return (p2[1]-p1[1]) / (p2[0]-p1[0])
    else:
        return None

'''
Find the intersection point of two **LINES**
'''
def getLineIntersect(l1,l2):
    p1, p2 = l1
    p3, p4 = l2
    (X1, Y1), (X2, Y2) = l1
    (X3, Y3), (X4, Y4) = l2

    s1 = slope(p1, p2)
    s2 = slope(p3, p4)

    # l1 and l2 are parallel
    if s1 == s2:
        b1, b2 = (None, None) if s1 is None else (Y1 - s1*X1, Y3 - s2*X3)
        if b1 == b2:
            # Parallel but same lime
            return (float('inf'),float('inf'))
        else:
            # Parallel but not same line
            return None
    # l1 and l2 are not vertical and not parellel
    elif (s1 is not None) and (s2 is not None):
        b1 = Y1 - s1*X1
        b2 = Y3 - s2*X3
        if s1 == s2:
            return (l1, l2)
        x = (b2 - b1) / (s1 - s2)
        y = x * s1 + b1
    # l1 is vertical
    elif s1 is None:
        b2 = Y3 - s2*X3
        x = X1
        y = x * s2 + b2
    # l2 is vertical
    else:
        b1 = Y1 - s1*X1
        x = X3
        y = x * s1 + b1

    return (x,y)

'''
Find if intersection point of line
'''
def calculateIntersect(ls1, ls2):
    # See if the infinite lines have a point in common
    p = getLineIntersect(ls1, ls2)
    if p == None:
        return None

    elif pointInLineSeg(ls1,ls2, p):
        return p
    else:
        return None

class Rador(object):
    def __init__(self, center, direction):
        self.center = center
        self.dir = np.array(direction)
        self.length = RADIUS

    def IntersectWall(self, wall, output):
        p1 = wall.pos
        p2 = np.array([wall.pos[0]+wall.width, wall.pos[1]])
        p3 = np.array([wall.pos[0]+wall.width, wall.pos[1]+wall.height])
        p4 = np.array([wall.pos[0], wall.pos[1]+wall.height])

        c1 = self.center
        c2 = np.array([self.center[0] + self.dir[0]*RADIUS, self.center[1] + self.dir[1]*RADIUS])

        i1 = calculateIntersect((c1,c2),(p1,p2))
        i2 = calculateIntersect((c1,c2),(p2,p3))
        i3 = calculateIntersect((c1,c2),(p3,p4))
        i4 = calculateIntersect((c1,c2),(p4,p1))
        output.put(i1)
        output.put(i2)
        output.put(i3)
        output.put(i4)

    def IntersectFrame(self, output):
        p1 = (0,0)
        p2 = (WIDTH,0)
        p3 = (WIDTH,HEIGHT)
        p4 = (0,HEIGHT)

        c1 = self.center
        c2 = np.array([self.center[0] + self.dir[0]*RADIUS, self.center[1] + self.dir[1]*RADIUS])
        i1 = calculateIntersect((c1,c2),(p1,p2))
        i2 = calculateIntersect((c1,c2),(p2,p3))
        i3 = calculateIntersect((c1,c2),(p3,p4))
        i4 = calculateIntersect((c1,c2),(p4,p1))
        output.append(i1)
        output.append(i2)
        output.append(i3)
        output.append(i4)
        return output

    def SearchIntersect(self, walls, out, i):
        output = mp.Queue()
        output2 = []

        processes = [mp.Process(target=self.IntersectWall, args=(w, output)) for w in walls]

        for p in processes:
            p.start()
        for p in processes:
            p.join()
        results = [output.get() for p in 4*processes]

        self.IntersectFrame(output2)
        res = list(filter(lambda x: not x is None, results))
        res2 = list(filter(lambda x: not x is None, output2))
        if len(res) == 0 and len(res2) == 0 :
            out.put((i, RADIUS))
        else:
            closest = float('inf')
            for r in res:
                dist = np.linalg.norm(r-self.center)
                if dist < closest:
                    closest = dist

            closest2 = float('inf')
            for r in res2:
                dist = np.linalg.norm(r-self.center)
                if dist < closest:
                    closest2 = dist
            out.put((i, min(closest, closest2)))

    def Update(self, center, direction):
        self.center = center
        self.dir = np.array(direction)


def createRador(angle,pos,direction):
    theta = np.radians(angle)
    c, s = np.cos(theta), np.sin(theta)
    R = np.matrix([[c, -s], [s, c]])
    head = np.squeeze(np.asarray(direction * R))
    return Rador(pos, head)


class Sensor(object):
    def __init__(self, pos, direction):
        self.center = pos
        self.dir = direction
        radors = [-16,-8,-4,0,4,8,16]
        self.radors = [createRador(r*THETA, self.center, self.dir) for r in radors]
        self.vector = []

    def search(self, walls):
        wallsC = [(i.pos[0] + i.width/2, i.pos[1]) for i in walls]

        relevant = []
        dists = [np.linalg.norm(self.center - i) for i in wallsC]
        for i, d in enumerate(dists):
            if d <= 100 + RADIUS:
                relevant.append(walls[i])

        output = mp.Queue()
        ''' return array of length 5 '''
        processes = [mp.Process(target=self.radors[i].SearchIntersect, args=(relevant, output, i)) for i in range(7)]
        for p in processes:
            p.start()
        for p in processes:
            p.join()
        results = [output.get() for p in processes]
        results.sort()
        self.vector = [x[1] for x in results]


    def updateAngle(self,angle):
        theta = np.radians(angle)
        c, s = np.cos(theta), np.sin(theta)
        R = np.matrix([[c, -s], [s, c]])
        head = np.squeeze(np.asarray(self.dir * R))
        return head

    def update(self, pos, direction, walls):
        self.center = pos
        self.dir = direction
        radors = [-16,-8,-4,0,4,8,16]
        angles = [self.updateAngle(i*THETA) for i in radors]
        for i, a in enumerate(angles):
            self.radors[i].Update(self.center, a)
        self.search(walls)

    def draw(self, screen):
        for i,r in enumerate(self.radors):
            pygame.draw.line(screen, (125,0,0), self.center, self.center + self.vector[i] * r.dir - 10)

if __name__ == '__main__':
    walls = []
    for i in range(WALL):
        walls.append(Wall(move = WALL_MOVE))
    wallsC = [(i.pos[0] + i.width/2, i.pos[1]) for i in walls]


    center = np.array([50., HEIGHT-50.])
    print()
    relevant = []
    for i, d in enumerate([np.linalg.norm(center - i) for i in wallsC]):
        if d <= 200:
            relevant.append(walls[i])

    direction = np.array([0.,-1.])
    s = Sensor(center, direction)
    start = time.time()
    results = s.search(relevant)
    end = time.time()
    print(s.vector)
    print(end - start)

    start = time.time()
    results = s.search(walls)
    end = time.time()
    print(s.vector)
    print(end - start)
