# coding: utf-8
from math import acos, asin
import numpy as np
from numpy.random import rand, uniform

class Particles(object):
    
    # class builder initiation 
    def __init__(self,L=5,N=300,eta=.02,r=1,dt=1,time=50,v=0.3,k=0,ep=1,*args,**kwargs):
        self.L = L #length of LxL domain
        self.N = N #number of particles
        self.eta = eta #noise for direction
        self.r = r #interaction radius
        self.dt = dt #timestep
        self.time = time #total iterations
        self.v = v #velocity
        self.ep = 1 
        #number of nearest neighbors or case number. 
        #If k = 0, this is the metric model. For k > 0, we use the top. model.
        if (type(k) == int):
            self.k = int(k)
        else:
            self.k = int(k)
            print('You did not provide an integer. To correct this, we have made your value of k = {0}'.format(self.k))
    # Set up initial condition
    def SetIC(self):
        posx = np.zeros((self.N,self.time))
        posy = np.zeros((self.N,self.time))
        velo = np.zeros((self.N,self.time))

        posx[:,0] = self.L*rand(self.N) #initial positions x-coordinates
        posy[:,0] = self.L*rand(self.N) #initial poisitions y-coordinates
        velo[:,0] = 2*np.pi*rand(self.N) #initial velocities
        return(posx,posy,velo)
    
    def ThetaNoise(self):
        #This is a uniform disribution of noise
        return(uniform(-self.eta/2,self.eta/2,self.N))
