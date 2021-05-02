from scipy import *
import numpy as np
import FlockingMethods_Numba as FMN 
from numba import int32, float32, float64, int64, njit, prange
from numpy.random import rand, uniform
import time 
import sys
 
@njit(parallel=True)
def Update_Metric(posx,posy,velo,Dists,r,ep,v,N,L,eta,dt):
    #This method allows us to update the flocking model with the Metric model
    #Each new velocity is constructed by averaging over all of the velocities within
    #the radius selected, self.r.
    #Inputs -- x-coordinates, y-coordinates, trajectories for time = t
    #Outputs -- x-coordinates, y-coordinates, trajectories for time = t + (delta t)
	
	Vals = np.zeros((N,N),dtype=int64)
	TotVals = np.zeros(N,dtype=int64)

	for j in prange(N):
		#find indicies that are within the radius
		#Vals = np.zeros((N,N),dtype=int64)
		#TotVals = np.zeros(N,dtype=int)
		for i in prange(N):
			if (Dists[j,i] <= r):
				Vals[j,TotVals[j]] = i
				TotVals[j] += 1
	sint = np.zeros(N,dtype=float32)
	cost = np.zeros(N,dtype=float32)
	avgc = np.zeros(N,dtype=float32)
	avgs = np.zeros(N,dtype=float32)

	for j in prange(N):
		#find average velocity of those inside the radius
		for k in prange(TotVals[j]):
			sint[j] += np.sin(velo[Vals[j,k]])
			cost[j] += np.cos(velo[Vals[j,k]])
		avgs[j] = sint[j]/TotVals[j]
		avgc[j] = cost[j]/TotVals[j]

	#construct the noise
	noise = uniform(-eta/2,eta/2,N)
	#update velocities and positions
	#print(len(velo))
	#print(len(avgc))
	#cosi = (ep)*avgc+(1-ep)*np.cos(velo)
	#sini = (ep)*avgs+(1-ep)*np.sin(velo)
	
	newvelo = np.zeros(N,dtype=float32)
	velon = np.zeros(N,dtype=float32)
	for j in prange(N):
		newvelo[j] = np.arctan2(avgs[j],avgc[j])
		velon[j] = np.mod(newvelo[j]+noise[j],2*np.pi)
		posx[j] += dt*v*np.cos(velon[j])
		posy[j] += dt*v*np.sin(velon[j]) 
	#newvelo = np.arctan2(sini,cosi) 
	#velon = np.mod(newvelo + noise,2*np.pi)
	#posx = posx + dt*v*np.cos(velon) 
	#posy = posy + dt*v*np.sin(velon)
	posx = np.mod(posx,L)
	posy = np.mod(posy,L)

	return(posx,posy,velon)

@njit(parallel=True)
def Calc_Dist(posx,posy,L):
	#find distance of every particle from particle j using periodic boundary conditions

	Dists = np.zeros((len(posx),len(posx)),dtype=float32)
	Dist0 = np.zeros((len(posx),len(posx)),dtype=float32)
	Dist1 = np.zeros((len(posx),len(posx)),dtype=float32)
	Dist2 = np.zeros((len(posx),len(posx)),dtype=float32)
	Dist3 = np.zeros((len(posx),len(posx)),dtype=float32)
	Dist4 = np.zeros((len(posx),len(posx)),dtype=float32)
	Dist5 = np.zeros((len(posx),len(posx)),dtype=float32)
	Dist6 = np.zeros((len(posx),len(posx)),dtype=float32)
	Dist7 = np.zeros((len(posx),len(posx)),dtype=float32)
	Dist8 = np.zeros((len(posx),len(posx)),dtype=float32)

	for j in prange(len(posx)):

		Dist0[:,j] = np.sqrt((posx[j] - posx)**2 + (posy[j] - posy)**2) #regular  
		Dist1[:,j] = np.sqrt((posx[j]  - L - posx)**2 + (posy[j] - posy + L)**2) #topleft
		Dist2[:,j] = np.sqrt((posx[j]  - posx)**2 + (posy[j] - posy + L)**2) #topcenter
		Dist3[:,j] = np.sqrt((posx[j]  + L - posx)**2 + (posy[j] - posy + L)**2) #topright
		Dist4[:,j] = np.sqrt((posx[j]  - L - posx)**2 + (posy[j] - posy)**2) #middleleft
		Dist5[:,j] = np.sqrt((posx[j]  + L - posx)**2 + (posy[j] - posy)**2) #middleright
		Dist6[:,j] = np.sqrt((posx[j]  - L - posx)**2 + (posy[j] - posy - L)**2) #bottomleft
		Dist7[:,j] = np.sqrt((posx[j]  - posx)**2 + (posy[j] - posy - L)**2) #bottomcenter
		Dist8[:,j] = np.sqrt((posx[j]  + L - posx)**2 + (posy[j] - posy - L)**2) #bottomright

		for k in prange(len(Dist0)):
			Dists[j,k] = min(Dist0[k,j],Dist1[k,j],Dist2[k,j],
				 	Dist3[k,j],Dist4[k,j],Dist5[k,j],
                		 	Dist6[k,j],Dist7[k,j],Dist8[k,j])
	return(Dists)

@njit
def Try_Go(posx,posy,velo,L,r,ep,v,N,eta,dt,timesteps):
	
	for d in range(1,timesteps):
		Dists = Calc_Dist(posx[:,d-1],posy[:,d-1],L)
		posx[:,d], posy[:,d], velo[:,d] = Update_Metric(posx[:,d-1],posy[:,d-1],velo[:,d-1],Dists,r,ep,v,N,L,eta,dt)
	
	return(posx,posy,velo)
#LL = int(sys.argv[2])
LL = 40
Noise = int(sys.argv[1])
NN = int(LL**2*.5)
timesteps = 6000

#Noises = [0,0.1,0.2,0.3,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.28]
Noises = [3.1,3.2,3.3,3.4,3.6,3.7,3.8,3.9]
ETA = Noises[Noise-1]
loops = 100
start = time.time()
for M in range(loops):
	if (M == 0):
		SM = FMN.Particles(L=LL,N=int(LL**2/2),eta=ETA,k = 0,r=3,dt=1,v=0.5,time=5,ep=1)
		posx, posy, velo = SM.SetIC()
		start1 = time.time()
		#posx, posy, velo = Try_Go(posx,posy,velo,SM.L,SM.r,SM.ep,SM.v,SM.N,SM.eta,SM.dt,timesteps)
		for d in range(1,SM.time):
			#posx[:,d],posy[:,d],velo[:,d] = Update_Metric(posx[:,d-1],posy[:,d-1],velo[:,d-1],
			 #                                    Dists,SM.r,SM.ep,SM.v,SM.N,SM.L,SM.eta,SM.dt)
			Dists = Calc_Dist(posx[:,d-1],posy[:,d-1],SM.L)
			posx[:,d],posy[:,d],velo[:,d] = Update_Metric(posx[:,d-1],posy[:,d-1],velo[:,d-1],
                                                     Dists,SM.r,SM.ep,SM.v,SM.N,SM.L,SM.eta,SM.dt)
		end1 = time.time()
		print('Dead Loop: {0} sec.'.format(end1-start1))

	SM = FMN.Particles(L=LL,N=int(LL**2/2),eta=ETA,k = 0,r=3,dt=1,v=0.5,time=timesteps,ep=1)
	posx, posy, velo = SM.SetIC()
	start1 = time.time()
	#posx, posy, velo = Try_Go(posx,posy,velo,SM.L,SM.r,SM.ep,SM.v,SM.N,SM.eta,SM.dt,timesteps)
	for d in range(1,SM.time):
		#posx[:,d],posy[:,d],velo[:,d] = Update_Metric(posx[:,d-1],posy[:,d-1],velo[:,d-1],
	         #                                    Dists,SM.r,SM.ep,SM.v,SM.N,SM.L,SM.eta,SM.dt)
		Dists = Calc_Dist(posx[:,d-1],posy[:,d-1],SM.L)
		posx[:,d],posy[:,d],velo[:,d] = Update_Metric(posx[:,d-1],posy[:,d-1],velo[:,d-1],
	                                             Dists,SM.r,SM.ep,SM.v,SM.N,SM.L,SM.eta,SM.dt)
	
	np.save('./NoiseSims/L40_X{0}_N{1}.npy'.format(M,ETA),posx)
	np.save('./NoiseSims/L40_Y{0}_N{1}.npy'.format(M,ETA),posy)
	np.save('./NoiseSims/L40_V{0}_N{1}.npy'.format(M,ETA),velo)
	end1 = time.time()

	print('Loop {0}: {1} sec. in {2} steps'.format(M+1,end1-start1,d))
end = time.time()
