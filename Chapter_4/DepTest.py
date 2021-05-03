from scipy import *
import numpy as np
import matplotlib.pyplot as plt
import PlanktonSignaling.basicsNumba2 as PS
import PlanktonSignaling.Deposition as DP
import profile
import copy
import time
import sys

import scipy.sparse as sp
from scipy.interpolate import RectBivariateSpline,griddata
from scipy.sparse.linalg import spsolve
from scipy.linalg import solve
from scipy.sparse import linalg as sla
from numpy.random import rand, uniform, triangular, choice
from scipy import sparse
from numba import int32, float32, float64, int64, njit, prange
from numpy import exp
from numba import prange

def initial_conditions(x,y):
    return(0*x + 0.01)

def scalarInterp(x_periodic,y_periodic,scalar_periodic,pos):
	bspline = RectBivariateSpline(x_periodic,y_periodic,scalar_periodic)
	return(bspline.ev(pos[:,1],pos[:,0]))

def scalarGrad(L,x_periodic,y_periodic,scalar_periodic,xp,dx=1.0e-4):
	dx = dx*L
	bspline = RectBivariateSpline(x_periodic,y_periodic,scalar_periodic)
	p = np.array([np.mod(xp + np.array([dx,0]),L),np.mod(xp - np.array([dx,0]),L),np.mod(xp + np.array([0,dx]),L),
                                  np.mod(xp - np.array([0,dx]),L)])

	dp = bspline.ev(p[:,:,1],p[:,:,0])

	diffs = np.array([dp[0]-dp[1],dp[2]-dp[3]])/2/dx
	diffs = diffs.T
	return(diffs)

@njit
def RT(k,delta,L,pos,vel,c,grad_c):
	# Actually, I need to do this as tumble and run, TR.
	for j in range(len(pos)):
		Dot = np.dot(vel[j],grad_c[j])
		alpha = 1/np.sqrt(delta**2 + Dot**2)
		if (rand() < k*0.5*(1-alpha*Dot)):
			th = rand()*2*pi
			vel[j] = np.array([np.cos(th),np.sin(th)])
	for j in range(len(pos)):
		pos[j] += k*vel[j]
		pos[j] = np.mod(pos[j],L)
	return(pos,vel)


@njit
def RTTK(k,delta,L,pos,vel,c,grad_c,Turn,tvv):
	for j in range(len(pos)):
		Dot = np.dot(vel[j].T,grad_c[j])
		alpha = 1/np.sqrt(delta**2 + Dot**2)
		if (rand() < k*0.5*(1- alpha*Dot)):
			th = np.arctan2(vel[j,1],vel[j,0])
			if (Turn == 0): #Uniform Dist. [0,2pi]
				th += rand()*2*pi
			elif (Turn == 1): #Uniform Dist. [a,b]
				Flip = np.random.choice(np.array([-1,1]))
				th += Flip*np.random.uniform(tvv[0],tvv[1])
			elif (Turn == 2 or Turn == 3): #Triangular Dist. 
				th += choice(np.array([-1,1]))*triangular(tvv[0],tvv[1],tvv[2])
			elif (Turn == 4):
				th += choice(np.array([-1,1]))*tvv[0]

			vel[j] = np.array([np.cos(th),np.sin(th)])
		pos[j] += k*vel[j]
		pos[j] = np.mod(pos[j],L)
	return(pos,vel)



@njit(parallel=True) 
def Update(N,L,k,Std,num,depStr,pos,xm,ym,intDelta,meshsize,boundaryCutoff):    
    
	f = np.zeros((N,N),dtype=float32)
	#f = np.zeros((N,N))
	for i in prange(num):
		Str = depStr[i]
		p = pos[i]
		A, B, C, D = 0,0,0,0
		centerX = int((meshsize-1)*p[0]/L+0.5)
		centerY = int((meshsize-1)*p[1]/L+0.5)
		lowerX      = max(0,centerX-intDelta)
		lowerXplus  = max(0,centerX-intDelta + (meshsize-1))
		lowerXminus = max(0,centerX-intDelta - (meshsize-1))
		upperX      = min(meshsize,centerX+intDelta)
		upperXplus  = min(meshsize,centerX+intDelta + (meshsize-1))
		upperXminus = min(meshsize,centerX+intDelta - (meshsize-1))
		lowerY      = max(0,centerY-intDelta)
		lowerYplus  = max(0,centerY-intDelta + (meshsize-1))
		lowerYminus = max(0,centerY-intDelta - (meshsize-1))
		upperY      = min(meshsize,centerY+intDelta)
		upperYplus  = min(meshsize,centerY+intDelta + (meshsize-1))
		upperYminus = min(meshsize,centerY+intDelta - (meshsize-1))
		sliceX = slice(lowerX,upperX+1)
		sliceY = slice(lowerY,upperY+1)
		f[sliceY,sliceX] += Str*(1/(4*pi*Std))*exp(-((xm[sliceY,sliceX]-p[0])**2
							   +(ym[sliceY,sliceX]-p[1])**2)/4/Std)
		if ((p[0])**2<boundaryCutoff):
			sliceX = slice(lowerXplus,upperXplus+1)
			sliceY = slice(lowerY,upperY+1)
			f[sliceY,sliceX] += Str*(1/(4*pi*Std))*exp(-((xm[sliceY,sliceX]-p[0]-L)**2
										   +(ym[sliceY,sliceX]-p[1])**2)/4/Std)
			A = 1
		if ((p[0]-L)**2<boundaryCutoff):
			sliceX = slice(lowerXminus,upperXminus+1)
			sliceY = slice(lowerY,upperY+1)
			f[sliceY,sliceX] += Str*(1/(4*pi*Std))*exp(-((xm[sliceY,sliceX]-p[0]+L)**2
								       +(ym[sliceY,sliceX]-p[1])**2)/4/Std)
			B = 1
		if ((p[1])**2<boundaryCutoff):
			sliceX = slice(lowerX,upperX+1)
			sliceY = slice(lowerYplus,upperYplus+1)
			f[sliceY,sliceX] += Str*(1/(4*pi*Std))*exp(-((xm[sliceY,sliceX]-p[0])**2
								    +(ym[sliceY,sliceX]-p[1]-L)**2)/4/Std)
			C = 1
		if ((p[1]-L)**2<boundaryCutoff):
			sliceX = slice(lowerX,upperX+1)
			sliceY = slice(lowerYminus,upperYminus+1)
			f[sliceY,sliceX] += Str*(1/(4*pi*Std))*exp(-((xm[sliceY,sliceX]-p[0])**2
							    +(ym[sliceY,sliceX]-p[1]+L)**2)/4/Std)
			D = 1
		if (A == 1 and C == 1): #Plankton in Lower Left Corner
			sliceX = slice(lowerXplus,upperXplus+1)
			sliceY = slice(lowerYplus,upperYplus+1)
			f[sliceY,sliceX] += Str*(1/(4*pi*Std))*exp(-((xm[sliceY,sliceX]-p[0]-L)**2
								 +(ym[sliceY,sliceX]-p[1]-L)**2)/4/Std)
		if (A == 1 and D == 1): #Plankton in Lower Left Corner
			sliceX = slice(lowerXplus,upperXplus+1)
			sliceY = slice(lowerYminus,upperYminus+1)
			f[sliceY,sliceX] += Str*(1/(4*pi*Std))*exp(-((xm[sliceY,sliceX]-p[0]-L)**2
								   +(ym[sliceY,sliceX]-p[1]+L)**2)/4/Std)
		elif (B == 1 and C == 1): #Plankton in Upper Right Corner
			sliceX = slice(lowerXminus,upperXminus+1)
			sliceY = slice(lowerYplus,upperYplus+1)
			f[sliceY,sliceX] += Str*(1/(4*pi*Std))*exp(-((xm[sliceY,sliceX]-p[0]+L)**2
								    +(ym[sliceY,sliceX]-p[1]-L)**2)/4/Std)
		elif (B == 1 and D == 1): #Plankton in Lower Right Corner
			sliceX = slice(lowerXminus,upperXminus+1)
			sliceY = slice(lowerYminus,upperYminus+1)
			f[sliceY,sliceX] += Str*(1/(4*pi*Std))*exp(-((xm[sliceY,sliceX]-p[0]+L)**2
									+(ym[sliceY,sliceX]-p[1]+L)**2)/4/Std)

	f = f.reshape((N**2,))
	return(f)


Dep = ['C','A','L']

Turn = int(sys.argv[2])

if (Turn == 0):
	tvv = [0]
if (Turn == 1):
	#Means = np.linspace(.75,np.pi-.75,10,dtype=float)
	#a = Means[int(sys.argv[3])-1]
	#tvv = np.array([a-.75,a+.75])
	Stds = np.linspace(.01,np.pi/3,6,dtype=float)
	a = Stds[int(sys.argv[3])-1]
	tvv = np.array([np.pi/3-a,np.pi/3+a])
if (Turn == 2):
	Med = np.linspace(0,1,11,dtype=float)*np.pi
	a = 0
	b = Med[int(sys.argv[3])-1]
	c = np.pi
	tvv = np.array([a,b,c])

if (Turn == 3):
	Med = np.arange(0,1,.1,dtype=float)*np.pi
	a = Med[int(sys.argv[3])-1]
	b = np.pi
	c = np.pi
	tvv = np.array([a,b,c])

if (Turn == 4):
	Means = np.linspace(0,np.pi,15,dtype=float)
	a = Means[int(sys.argv[3])-1]
	tvv = np.array([a])

TKS = ['Random','UnifSPi3_','Triangle','TriangleA','SingleAngle']
meshsize = 400  #Chemical Mesh size
numb = 160000  #Number of plankton in simulation
LL = 10 #Length of domain [0,L] x [0,L]
dt = 0.01 #Time-stepping size
TotalTime = float(sys.argv[1]) #Total time 
simTime = int(TotalTime/dt) #Number of timesteps in order to achieve total Time 

Start = int(sys.argv[4])
End = int(sys.argv[5])

for Job in range(Start,End+1):

	j = int(sys.argv[3])

	if (j == 1):
		SM = PS.Plankton(DP.constantDep,d1=.1,d2=4,N = meshsize,depMaxStr=.01,
				    Const=3,L=LL,k=dt,delta=1e-3,depThreshold=0.012, 
				depTransWidth=0.008, num = numb, c0=0.01)
	if (j == 2):
		SM = PS.Plankton(DP.atanDep,d1=.1,d2=4,N = meshsize,depMaxStr=.01,
				   Const=3,L=LL,k=dt,delta=1e-3,depThreshold=0.012, 
				  depTransWidth=0.0007, num = numb, c0=0.01)

	if (j == 3):
		SM = PS.Plankton(DP.linAtanDep,d1=.1,d2=4,N = meshsize,depMaxStr=.01,
				   Const=3,L=LL,k=dt,delta=1e-3,depThreshold=0.012, 
				   depTransWidth=0.008, num = numb, c0=0.01)

	SM.SetIC(initial_conditions)

	lenn = int(np.sqrt(numb))
	pos = np.zeros((1,2))
	vel = np.zeros((1,2))

	#Place plankton down uniformly throughout the domain and give each a direction to travel initially
	for l in range(0,lenn):
		for k in range(0,lenn):
			pos = np.append(pos,[np.array([np.mod(k*(SM.L*1/(lenn)) + 0.5*(SM.L*1/(lenn)),SM.L),
					np.mod(l*(SM.L*1/(lenn)) + 0.5*(SM.L*1/(lenn)),SM.L)])],axis=0)
			th  = rand()*2*pi
			vel = np.append(vel,[np.array([np.cos(th),np.sin(th)])],axis=0)

	pos = np.delete(pos,0,0)
	vel = np.delete(vel,0,0)
	pos_store = list([pos[:,:]])
	pos_store = list([np.array(pos)])
	scalar_store = list([SM.Meshed()])

	CHEM = np.zeros((2,meshsize,meshsize))
	POS = np.zeros((2,numb,2))
	CHEM[1,:,:] = scalar_store[0] #preallocate the chemical 
	POS[1,:,:] = pos_store[0] #preallocate the plankton
	Dep = ['C','A','L']
	MaxChem = np.zeros(simTime)
	MinChem = np.zeros(simTime)
	TotalGrad = np.zeros(simTime)
	TotChem = np.zeros(simTime)

	MaxChem[0] = max(CHEM[1].flatten())
	MinChem[0] = min(CHEM[1].flatten())
	TotalGrad[0] = np.sum(np.absolute(np.gradient(CHEM[1],LL/meshsize)))*(LL/meshsize)**2
	TotChem[0] = np.sum(CHEM[1].flatten())*(LL/meshsize)**2
	
	boundaryCutoff = 64*SM.depVar
	intDelta = int((SM.N - 1)*8*np.sqrt(SM.depVar)/SM.L+0.5)
	Std = SM.depVar
	meshsize = SM.N
	PlankDensity = SM.density*SM.L**2/SM.num

	LU = sla.splu(SM.M1.tocsc())

	Time1 = np.zeros(simTime-1)
	Time2 = np.zeros(simTime-1)
	Time3 = np.zeros(simTime-1)
	Time4 = np.zeros(simTime-1)
	Time5 = np.zeros(simTime-1)
	Time = np.zeros(simTime-1)
	Times = np.linspace(0,simTime,21,dtype=int)

	print('Starting Numba Version.....')
	print('Meshsize: {0}'.format(meshsize))
	print('Numb. of Plank: {0}'.format(numb))
	for k in range(1,simTime):
		start = time.time()

		start1 = time.time()
		SM.BuildPeriodic()
		c = scalarInterp(SM.x_periodic,SM.y_periodic,SM.scalar_periodic,pos)
		SM.BuildPeriodic()
		grad_c = scalarGrad(SM.L,SM.x_periodic,SM.y_periodic,SM.scalar_periodic,pos)
		end1 = time.time()

		start2 = time.time()
		pos,vel = RTTK(SM.k,SM.delta,SM.L,pos,vel,c,grad_c,Turn,tvv)
		end2 = time.time()

		start3 = time.time()
		depStr = SM.depFcn(c,SM.depMaxStr,SM.depThreshold,SM.depTransWidth)
		end3 = time.time()
	    
		start4 = time.time()
		f = Update(SM.N,SM.L,SM.k,Std,SM.num,depStr,pos,SM.xm,SM.ym,intDelta,meshsize,boundaryCutoff)
		end4 = time.time()

		start5 = time.time()
		SM.scalar = LU.solve(SM.M2.dot(SM.scalar)+SM.k*(PlankDensity)*f)
		end5 = time.time()

		end = time.time()

		Time1[k-1] = end1-start1
		Time2[k-1] = end2-start2
		Time3[k-1] = end3-start3
		Time4[k-1] = end4-start4
		Time5[k-1] = end5-start5

		Time[k-1] = end-start

		CHEM[0,:,:] = CHEM[1]
		CHEM[1,:,:] = SM.Meshed()
		POS[0,:,:] = POS[1]
		POS[1,:,:] = pos
		MaxChem[k] = max(CHEM[1].flatten())
		MinChem[k] = min(CHEM[1].flatten())
		TotChem[k] = np.sum(CHEM[1].flatten())*(LL/meshsize)**2
		#TotalGrad[k] = np.sum(np.absolute(np.gradient(CHEM[1],LL/meshsize)))*(LL/meshsize)**2
		A, B = np.gradient(CHEM[1],LL/meshsize)
		TotalGrad[k] = np.sum(np.sqrt(A**2 + B**2))*(LL/meshsize)**2
		if (k in Times):
			np.save('./DepositionTests/Pos_Dep{1}_T{0}_J{2}'.format(int(k*SM.k),Dep[j-1],Job),pos)
			np.save('./DepositionTests/Chem_Dep{1}_T{0}_J{2}'.format(int(k*SM.k),Dep[j-1],Job),CHEM[1])
			print('Loop {0}: {1} sec.'.format(k,end-start))


	print('Time 1: {0}'.format(round(mean(Time1[1:]),10)))
	print('Time 2: {0}'.format(round(mean(Time2[1:]),10)))
	print('Time 3: {0}'.format(round(mean(Time3[1:]),10)))
	print('Time 4: {0}'.format(round(mean(Time4[1:]),10)))
	print('Time 5: {0}'.format(round(mean(Time5[1:]),10)))
	#print('Total Chemical = {0}'.format(TotalChem))
	np.save('./DepositionTests/Pos_Dep{1}_T{0}_J{2}'.format(TotalTime,Dep[j-1],Job),pos)
	np.save('./DepositionTests/Chem_Dep{1}_T{0}_J{2}'.format(TotalTime,Dep[j-1],Job),CHEM[1])
	np.save('./DepositionTests/MaxMin_Dep{1}_T{0}_J{2}'.format(TotalTime,Dep[j-1],Job),MaxChem-MinChem)
	np.save('./DepositionTests/TotGrad_Dep{1}_T{0}_J{2}'.format(TotalTime,Dep[j-1],Job),TotalGrad)
	np.save('./DepositionTests/TotChem_Dep{1}_T{0}_J{2}'.format(TotalTime,Dep[j-1],Job),TotChem)

