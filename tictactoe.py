#!/usr/bin/env python

# Author: Team3
# Date: 2024-03-08

#Random Empty File. 
#! /usr/bin/env python



import numpy as np

import random


def subArr(A,B):
	if(A==B):
		return True
	
	for i in range(len(A) - len(B)):
		if A[i:i+len(B)] == B:
			return True

#array corresponding to played suares
#0=empty, 1=player, 2=robot
def play(mat):
 	
	#just brute force a match. feel free to replace this with a smarter algorithm.
	#only for cases where player or robot is about to win (2 of same number in a row/column/cross)
	if(subArr(mat[0:3],[1,1,0]) or subArr(mat[0:3],[2,2,0])):
		return 2
	if(subArr(mat[0:3],[1,0,1]) or subArr(mat[0:3],[2,2,0])):
		return 1
	if(subArr(mat[0:3],[0,1,1]) or subArr(mat[0:3],[2,2,0])):
		return 0
	if(subArr(mat[3:6],[1,1,0]) or subArr(mat[0:3],[2,2,0])):
		return 5
	if(subArr(mat[3:6],[1,0,1]) or subArr(mat[0:3],[2,2,0])):
		return 4
	if(subArr(mat[3:6],[0,1,1]) or subArr(mat[0:3],[2,2,0])):
		return 3
	#etc
	#for columns:
	if(subArr([mat[0], mat[3], mat[6]],[1,1,0]) or subArr(mat[0:3],[2,2,0])):
		return 999
	#etc

	#if no close-win scenario, pick a random spot
	r = random.random(0,9)
	while(mat[r] is not 0):
		r = random.random(0,9) #keep picking until spot is empty
	
	return r