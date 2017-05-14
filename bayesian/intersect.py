import numpy as np
import math
from variables import *

''' 
Determines whether the box representing two line segments cross
'''
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

if __name__ == "__main__":

    # line 1 and 2 cross, 1 and 3 don't but would if extended, 2 and 3 are parallel
    # line 5 is horizontal, line 4 is vertical
	p1 = (1.,5.)
	p2 = (4.,7.)

	p3 = (4.,5.)
	p4 = (3.,7.)

	p5 = (4.,1)
	p6 = (3.,3.)

	p7 = (3,1)
	p8 = (3,10)

	p9 =  (0.,6.)
	p10 = (5.,6.)

	p11 = (472.0, 116.0)
	p12 = (542.0, 116.0)  

	assert None != calculateIntersect((p1, p2), (p3, p4)), "line 1 line 2 should intersect"
	assert None != calculateIntersect((p3, p4), (p1, p2)), "line 2 line 1 should intersect"
	assert None == calculateIntersect((p1, p2), (p5, p6)), "line 1 line 3 shouldn't intersect"
	assert None == calculateIntersect((p3, p4), (p5, p6)), "line 2 line 3 shouldn't intersect"
	assert None != calculateIntersect((p1, p2), (p7, p8)), "line 1 line 4 should intersect"
	assert None != calculateIntersect((p7, p8), (p1, p2)), "line 4 line 1 should intersect"
	assert None != calculateIntersect((p1, p2), (p9, p10)), "line 1 line 5 should intersect"
	assert None != calculateIntersect((p9, p10),( p1, p2)), "line 5 line 1 should intersect"
	assert None != calculateIntersect((p7, p8), (p9, p10)), "line 4 line 5 should intersect"
	assert None != calculateIntersect((p9, p10),( p7, p8)), "line 5 line 4 should intersect"

	print ("\nSUCCESS! All asserts passed for doLinesIntersect")

