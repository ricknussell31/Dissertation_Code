from scipy import *
import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
import matplotlib.animation
import matplotlib.animation as animation

# ## This modules defined all of the animation and plotting routines that are necessary ###

def PlotSOCombined(Class,ck,pk,time):
    
    #Plots the chemical and plankton for the second order problem for a given time#
    
    c, p = CenterData(Class,ck[time,:],pk[time,:])

    fig, ax1 = plt.subplots() #put rho and c on the same figure
    
    ax1.set_xlabel(r'$x$')
    ax1.set_ylabel(r'$\rho$: Plankton Density', color='red')

    ax1.set_xlim(Class.left, Class.right)
    ax1.plot(Class.xm,p,color='red', label='Plankton, T = {0}'.format(round(Class.dt*time)))

    ax2 = ax1.twinx()  #create a second axes that shares the same x-axis
    ax2.set_ylabel(r'$c$: Chemical Concentration', color='blue')
    ax2.plot(Class.xm, c, color='blue', label='Chemical, T = {0}'.format(round(Class.dt*time)))
    
    r = str(Class.depFcn).replace("function", "")

    #plt.title(r'Second Order, $d_1$: {0}, $d_2$: {1}, $\delta$: {2}, {3}'.format(round(Class.d1,2), round(Class.d2,2), Class.delta, r[2:9]))
    plt.title(r'Time: {3}, $d_1$: {0}, $d_2$: {1}, $\delta$: {2}'.format(round(Class.d1,2), round(Class.d2,2), Class.delta, round(Class.dt*time,2)))
    
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    #ax2.legend(lines + lines2, labels + labels2, loc=0)

    #plt.legend(loc='best')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped 
    plt.show()

def PlotFOCombined(Class,ck,pk,qk,time):
    
    #Plots the chemical and plankton for the first order problem for a given time#
        
    c, p = CenterData(Class,ck[time,:],pk[time,:]+qk[time,:])
    
    
    fig, ax1 = plt.subplots() #put rho and c on the same figure
    
    ax1.set_xlabel(r'$x$')
    ax1.set_ylabel(r'$\rho$: Plankton Density', color='red')

    ax1.set_xlim(Class.left, Class.right)
    ax1.plot(Class.xm,p,color='red', label='Plankton, T = {0}'.format(round(Class.dt*time)))

    ax2 = ax1.twinx()  #create a second axes that shares the same x-axis
    ax2.set_ylabel(r'$c$: Chemical Concentration', color='blue')
    ax2.plot(Class.xm, c, color='blue', label='Chemical, T = {0}'.format(round(Class.dt*time)))
    
    r = str(Class.depFcn).replace("function", "")

    plt.title(r'First Order, $d_1$: {0}, $d_2$: {1}, $\delta$: {2}, {3}'.format(round(Class.d1,2), round(Class.d2,2), Class.delta, r[2:9]))
    
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=0)

    #plt.legend(loc='best')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped 
    plt.show()

def MultipleTimesPSO(Class,pk,times):
    
    #Plots the plankton density evolution for a given array of times for the second order equation#
    
    plt.figure()
    T = len(times)
    for i in range(0,T):
        plt.plot(Class.xm,pk[times[i]],label='T = {0}'.format(times[i]*Class.dt))
    
    r = str(Class.depFcn).replace("function", "")
    
    plt.legend(loc=0)
    plt.xlabel(r'x')
    plt.ylabel(r'Total Plankton')
    plt.title('Plankton, Second Order, $d_1$: {0}, $d_2$: {1}, $\delta$: {2}, {3}'.format(round(Class.d1,2), round(Class.d2,2), Class.delta, r[2:9]))
    plt.show()

def MultipleTimesPFO(Class,rk,qk,times):
    
    #Plots the plankton density evolution for a given array of times for the first order equation#

    plt.figure()
    pk = rk + qk
    T = len(times)
    for i in range(0,T):
        plt.plot(Class.xm,pk[times[i]],label='T = {0}'.format(times[i]*Class.dt))
    
    plt.legend(loc=0)
    
    r = str(Class.depFcn).replace("function", "")

    plt.xlabel(r'x')
    plt.ylabel(r'Total Plankton')
    plt.title('Plankton, First Order, $d_1$: {0}, $d_2$: {1}, $\delta$: {2}, {3}'.format(round(Class.d1,2), round(Class.d2,2), Class.delta, r[2:9]))
    plt.show()

def MultipleTimesCSO(Class,ck,times):
    
    #Plots the chemical concentration evolution for a given array of times for the second order equation#
        
    plt.figure()
    T = len(times)
    for i in range(0,T):
        plt.plot(Class.xm,ck[times[i]],label='T = {0}'.format(times[i]*Class.dt))
    
    r = str(Class.depFcn).replace("function", "")
    
    plt.legend(loc=0)
    plt.xlabel(r'x')
    plt.ylabel(r'Chemical Concentration')
    plt.title('Chemical, Second Order, $d_1$: {0}, $d_2$: {1}, $\delta$: {2}, {3}'.format(round(Class.d1,2), round(Class.d2,2), Class.delta, r[2:9]))
    plt.show()

def MultipleTimesCFO(Class,ck,times):
    
    #Plots the chemical concentration evolution for a given array of times for the first order equation#

    plt.figure()
    T = len(times)
    for i in range(0,T):
        plt.plot(Class.xm,ck[times[i]],label='T = {0}'.format(times[i]*Class.dt))
    
    plt.legend(loc=0)
    
    r = str(Class.depFcn).replace("function", "")

    plt.xlabel(r'x')
    plt.ylabel(r'Chemical Concentration')
    plt.title('Chemical, First Order, $d_1$: {0}, $d_2$: {1}, $\delta$: {2}, {3}'.format(round(Class.d1,2), round(Class.d2,2), Class.delta, r[2:9]))
    plt.show()

def CenterData(Class,c,p):
    
    #Centers the data for plotting so that the maximum value is centered.#
    
    RhoNew = 0*p
    CNew = 0*p

    A = max(p)
    for i in range(0,len(Class.xm)):
        if (p[i] == A):
            K = i

    Ha = int((Class.N+1)/2)
    dist = abs(K - Ha)
    if (K < Ha):
        for i in range(0,len(Class.xm)):
            RhoNew[(i+dist)%Class.N] = p[i]
            CNew[(i+dist)%Class.N] = c[i] 
        RhoNew[len(RhoNew)-1] = RhoNew[0]
        CNew[len(CNew)-1] = CNew[0]
    elif (K > Ha):
        for i in range(0,len(Class.xm)):
            RhoNew[(i-dist)%Class.N] = p[i]
            CNew[(i-dist)%Class.N] = c[i]
        RhoNew[len(RhoNew)-1] = RhoNew[0]
        CNew[len(CNew)-1] = CNew[0]
    elif (K == Ha):
        RhoNew = p
        CNew = c
    return(CNew,RhoNew)

def TotalsSO(Class,c,p):
        plt.figure()
        time2 = linspace(0,len(p)*Class.dt,len(p))
        plt.plot(time2,(p/p[0])*100,label='Plankton')
        plt.plot(time2,(c/c[0])*100,label='Chemical')
        plt.xlabel('Time')
        plt.ylabel('Percentage Left')
        plt.title('Total Plankton/Chemical Over Time, Second Order')
        
        plt.show()

def TotalsFO(Class,c,p,q):
        plt.figure()
        time2 = linspace(0,len(p)*Class.dt,len(p))
        plt.plot(time2,((p+q)/(p[0] + q[0]))*100,label='Plankton')
        plt.plot(time2,(c/c[0])*100,label='Chemical')
        plt.xlabel('Time')
        plt.ylabel('Percentage Left')
        plt.title('Total Plankton/Chemical Over Time, First Order')
        
        plt.show()

def AnimateThisSO(Class,c,p,name,TT):

    fig, ax1 = plt.subplots()
    time = 0

    line, = ax1.plot(Class.xm,p[time,:], color='blue', label='Time = {0}'.format(round(time*Class.dt,3)))
    ax1.set_xlabel(r'$x$')
    ax1.set_ylabel(r'Plankton', color='blue')
    ax1.set_xlim(Class.left, Class.right)

    ax2 = ax1.twinx()  #create a second axes that shares the same x-axis
    line2, = ax2.plot(Class.xm, c[time,:], color='red', label='Time = {0}'.format(round(time*Class.dt,3)))
    ax2.set_ylabel(r'Chemical Concentration', color='red')
    ax2.set_xlim(Class.left, Class.right)
    fig.tight_layout()


    def init():  # only required for blitting to give a clean slate.
        line.set_ydata([np.nan] * len(Class.xm))
        line2.set_ydata([np.nan] * len(Class.xm))
        return line,


    def animate(i):
        line.set_ydata(p[i,:])  # update the data.
        M = max(p[i,:])
        m = min(p[i,:])

        if (M-m != 0):
            ax1.set_ylim(m-1, M+1)
            
        line2.set_ydata(c[i,:])  # update the data.

        M = max(c[i,:])
        m = min(c[i,:])
        if (M-m != 0):
            ax2.set_ylim(m -.0001, M +0.0001)
            
        r = str(Class.depFcn).replace("function", "")

        plt.title(r'Plankton/Chemical, Time = {3}, $d_1$: {0}, $d_2$: {1}, $N$: {2}, Func: {4}'.format(round(Class.d1,2), round(Class.d2,2), Class.N, round(i*Class.dt,3),r[2:9]))
        return line, line2,


    ani = animation.FuncAnimation(
        fig, animate, init_func=init, interval=1, blit=True,save_count=TT-1)


    ani.save("{0}.mp4".format(str(name)))
