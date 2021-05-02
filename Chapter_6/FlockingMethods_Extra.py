# coding: utf-8

from scipy import *
from math import acos, asin
import scipy.sparse as sp
import numpy as np
from scipy.sparse.linalg import spsolve
from numpy.random import rand, uniform

class Background_Field(object):
    
    # class builder initiation 
    def __init__(self,L=5,N=300,eta=.02,r=1,dt=1,time=50,*args,**kwargs):
        self.L = L #length of LxL domain
        self.N = N #number of particles
        self.eta = eta #noise for direction
        self.r = r #interaction radius
        self.dt = dt #timestep
        self.time = time #total iterations
        
    # Set up initial condition
    def SetIC(self):
        posx = np.zeros((self.N,self.time))
        posy = np.zeros((self.N,self.time))
        velo = np.zeros((self.N,self.time))

        posx[:,0] = self.L*rand(self.N) #initial positions x-coordinates
        posy[:,0] = self.L*rand(self.N) #initial poisitions y-coordinates
        velo[:,0] = 2*pi*rand(self.N) #initial velocities
        return(posx,posy,velo)
    
    def ThetaNoise(self):
        return(uniform(-self.eta/2,self.eta/2,self.N))
    
class Particles(Background_Field):
    
    def __init__(self,v=0.3, ep=1, *args,**kwargs):

        self.v = v #constant speed
        self.ep = ep #new parameter
        self.args = args
        self.kwargs = kwargs
        
        super(Particles,self).__init__(*args,**kwargs)
         
    def Update_Rad(self,posx,posy,velo):
        avgs = 0*velo
        avgc = 0*velo
        L = self.L
        for j in range(0,self.N):
            #find distance using periodic boundary conditions
            Dist0 = sqrt((posx[j] - posx)**2 + (posy[j] - posy)**2) #regular  
            Dist1 = sqrt((posx[j]  - L - posx)**2 + (posy[j] - posy + L)**2) #topleft
            Dist2 = sqrt((posx[j]  - posx)**2 + (posy[j] - posy + L)**2) #topcenter
            Dist3 = sqrt((posx[j]  + L - posx)**2 + (posy[j] - posy + L)**2) #topright
            Dist4 = sqrt((posx[j]  - L - posx)**2 + (posy[j] - posy)**2) #middleleft
            Dist5 = sqrt((posx[j]  + L - posx)**2 + (posy[j] - posy)**2) #middleright
            Dist6 = sqrt((posx[j]  - L - posx)**2 + (posy[j] - posy - L)**2) #bottomleft
            Dist7 = sqrt((posx[j]  - posx)**2 + (posy[j] - posy - L)**2) #bottomcenter
            Dist8 = sqrt((posx[j]  + L - posx)**2 + (posy[j] - posy - L)**2) #bottomright
            TD = [Dist0,Dist1,Dist2,Dist3,Dist4,Dist5,Dist6,Dist7,Dist8]
            Dist = np.asarray(TD).min(0) #minimum values for all possible distances
            
            
            Vals = [i for i in range(len(Dist)) if Dist[i] <= self.r] #indices of all objects in the radius
            
            #find average velocity
            sint = 0 
            cost = 0
            for k in Vals:
                sint = sint + sin(velo[k])
                cost = cost + cos(velo[k])
            avgs[j] = sint/len(Vals)
            avgc[j] = cost/len(Vals)

        noise = self.ThetaNoise()
        
        #update velocities and positions
    
        cosi = (self.ep)*avgc+(1-self.ep)*cos(velo)
        sini = (self.ep)*avgs+(1-self.ep)*sin(velo)
        newvelo = np.arctan(np.divide(sini,cosi))
        for n in range(0,len(newvelo)):
            if (sini[n] < 0 and cosi[n] < 0):
                newvelo[n] = mod(newvelo[n]+pi, 2*pi)
            if (sini[n] > 0 and cosi[n] < 0):
                newvelo[n] = mod(newvelo[n]+pi, 2*pi)
        velon = mod(newvelo + noise,2*pi)
        posx = posx + self.dt*self.v*cos(velon) 
        posy = posy + self.dt*self.v*sin(velon)
        
        posx,posy = self.CheckBoundary(posx,posy)
        return(posx,posy,velon)
    
    def Update_Top(self,posx,posy,velo,kn):
        #k here is the number of nearest neighbors selected to average over
        avgs = 0*velo
        avgc = 0*velo
        L = self.L
        for j in range(0,self.N):
            
            Dist = self.Calc_Dist(posx,posy,L,j)
            
            #Find k-nearest neighbors
            
            Vals = Dist.argsort()[:kn] #indices of the k-nearest neighbors
            #print('Vals = {0}'.format(len(Vals)))
            #find average velocity
            sint = 0 
            cost = 0
            for k in Vals:
                sint = sint + sin(velo[k])
                cost = cost + cos(velo[k])
            avgs[j] = sint/len(Vals)
            avgc[j] = cost/len(Vals)

        noise = self.ThetaNoise()
        
        #update velocities and positions
    
        cosi = (self.ep)*avgc+(1-self.ep)*cos(velo)
        sini = (self.ep)*avgs+(1-self.ep)*sin(velo)
        newvelo = np.arctan(np.divide(sini,cosi))
        for n in range(0,len(newvelo)):
            if (sini[n] < 0 and cosi[n] < 0):
                newvelo[n] = mod(newvelo[n]+pi, 2*pi)
            if (sini[n] > 0 and cosi[n] < 0):
                newvelo[n] = mod(newvelo[n]+pi, 2*pi)
        velon = mod(newvelo + noise,2*pi)
        posx = posx + self.dt*self.v*cos(velon) 
        posy = posy + self.dt*self.v*sin(velon)
        
        posx,posy = self.CheckBoundary(posx,posy)
        return(posx,posy,velon)
    
    
    def Calc_Dist(self,posx,posy,L,j):
        #find distance of every particle from particle j using periodic boundary conditions
        
        Dist0 = sqrt((posx[j] - posx)**2 + (posy[j] - posy)**2) #regular  
        Dist1 = sqrt((posx[j]  - L - posx)**2 + (posy[j] - posy + L)**2) #topleft
        Dist2 = sqrt((posx[j]  - posx)**2 + (posy[j] - posy + L)**2) #topcenter
        Dist3 = sqrt((posx[j]  + L - posx)**2 + (posy[j] - posy + L)**2) #topright
        Dist4 = sqrt((posx[j]  - L - posx)**2 + (posy[j] - posy)**2) #middleleft
        Dist5 = sqrt((posx[j]  + L - posx)**2 + (posy[j] - posy)**2) #middleright
        Dist6 = sqrt((posx[j]  - L - posx)**2 + (posy[j] - posy - L)**2) #bottomleft
        Dist7 = sqrt((posx[j]  - posx)**2 + (posy[j] - posy - L)**2) #bottomcenter
        Dist8 = sqrt((posx[j]  + L - posx)**2 + (posy[j] - posy - L)**2) #bottomright
        
        TD = [Dist0,Dist1,Dist2,Dist3,Dist4,Dist5,Dist6,Dist7,Dist8]
        
        return(np.asarray(TD).min(0)) #minimum values for all possible distances
    
    def CheckBoundary(self,posx,posy):
        xcordn = [i for i in range(self.N) if posx[i] < 0]
        xcordp = [i for i in range(self.N) if posx[i] > self.L]
        ycordn = [i for i in range(self.N) if posy[i] < 0]
        ycordp = [i for i in range(self.N) if posy[i] > self.L]
        
        for j in xcordn:
            posx[j] = posx[j] + self.L
       
        for j in xcordp:
            posx[j] = posx[j] - self.L
            
        for j in ycordn:
            posy[j] = posy[j] + self.L
            
        for j in ycordp:
            posy[j] = posy[j] - self.L
           
        return(posx,posy)
                                      