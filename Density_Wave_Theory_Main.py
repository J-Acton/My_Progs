# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 11:07:41 2023

Description:
    A program to simulate a spiral arm galaxy by setting up tilted elliptical 
    orbits, populated by stars, which then illustrate the origin of the spiral
    arms once the model is animated.


@author: James Acton (20325303)
"""

#-importing libraries-#
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from DWT_Funcs import Galaxy, calculations, anim_vals, bright_distrib, H_II


#setting initial constants
N = 35000 #number of stars
r_core = 15/64 #galaxy core radius
r_galax = 15000 #galaxy radius
r_dist = 2*r_galax
e1 = 0.8 #inner ellipticity
e2 = 1.0 #outer ellipticity
ang_offset = 0.0004
vel = 5e-6 #velocity of stars
t_step = 0.25e4 #timestep

St_Spiral = Galaxy(N, r_core, r_galax, e1, e2, ang_offset, vel, t_step) #defining galaxy object from Galaxy.py


x, y, angles, a, tilts, b = calculations(St_Spiral.N, St_Spiral.r_core,      #Using the calculations function from Galaxy.py to calculate the initial x and
                                         St_Spiral.r_galax, St_Spiral.e1,    #y coords for each star in St_Spiral, as well as the semi-major and semi-minor
                                         St_Spiral.e2, St_Spiral.ang_offset) #axes for each stars elliptical orbit and the orbital tilt of the elliptical orbit


Intensity = bright_distrib(x, y, St_Spiral.r_core, St_Spiral.r_galax) #Using the bright_distrib function from Galaxy.py to generate a brightness distribution
I = Intensity/(np.max(Intensity)) #normalising the intensity values so they can be applied as alpha values when plotting


fig=plt.figure(figsize=(50, 28)) #creating figure and setting dimensions
ax = plt.gca()
ax.set_facecolor('#01021D') #setting background colour of plot to look space-like
ax.set_title('Spiral Galaxy Simulation')

frms = 250 #setting the number of frames for animating
intrvl=(1e4)/144 #setting framerate

an_x, an_y = anim_vals(x, y, angles, frms, vel, t_step, a, tilts, b) #using the anim_vals function from galaxy.py to generate arrays of x and y values for each star


initial = ax.scatter(an_x[0], an_y[0], alpha=I, marker='o', s=0.25, color='w') #setting the initial frame

#setting the initial values for the 4 ellipse sections used to plot HII region
u1, v1, a1, b1 = -750, -950, 3000, 2400 #u=x-pos, v=y-pos, a=semi-major radius, b=semi-minor radius
u2, v2, a2, b2 = 1775, 1750, 7200, 6000
u3, v3, u4, v4 = -750, -950, 1750, 1750
st1, mid1, end1 = 0.610865, 3.92699, 6.8*np.pi/4 #range of angle values for which HII regions appear
p, q = 1.375, 2

an_ell1, an_ell2 = H_II(30, u1, v1, a1, b1, u2, v2, a2, b2, st1, mid1, end1, 
                        frms)
an_ell3, an_ell4 = H_II(30, u3, v3, a1, b1, u4, v4, a2, b2, st1, mid1, end1, 
                        frms)

#generating initial plots for all ellipse sections
def H_init(u, v, a, b, p, q, ellip):
    initial = ax.scatter(p*(u+a*np.cos(ellip)), q*(v+b*np.sin(ellip)), 
                         alpha=0.45, marker='o', s=2.5, color='r')
    return initial
    
initial2 = H_init(u1, v1, a1, b1, p, -q, an_ell1[0]) 
initial3 = H_init(u2, v2, a2, b2, p, -q, an_ell2[0]) 
initial4 = H_init(u3, v3, a1, b1, -p, q, an_ell3[0]) 
initial5 = H_init(u4, v4, a2, b2, -p, q, an_ell4[0]) 

#animation functions for the galaxy, as well as the sections of the HII regions

def animate(i, an_x, an_y, initial):  
    xi = an_x[i]
    yi = an_y[i]
    initial.set_offsets(np.c_[xi, yi])
    return initial

def anim_func(Figure, an_x, an_y, initial):
    return FuncAnimation(Figure, animate, frames=frms, fargs=(an_x, an_y, initial), interval = intrvl, repeat=True)


#animating the entire plot
anim1=anim_func(fig, an_x, an_y, initial)
anim2=anim_func(fig, 1.375*(u1+a1*np.cos(an_ell1)), -2*(v1+b1*np.sin(an_ell1)), 
                initial2)
anim3=anim_func(fig, 1.375*(u2+a2*np.cos(an_ell2)), -2*(v2+b2*np.sin(an_ell2)),
                initial3)
anim4=anim_func(fig, -1.375*(u3+a1*np.cos(an_ell3)), 2*(v3+b1*np.sin(an_ell3)),
                initial4)
anim5=anim_func(fig, -1.375*(u4+a2*np.cos(an_ell4)), 2*(v4+b2*np.sin(an_ell4)), 
                initial5)
plt.show()

"""
#lines for saving animatd plot as a gif.
writer = PillowWriter(fps=60, metadata=dict(artist='James Acton'), bitrate=1800)
anim1.save('Final_Galaxy.gif')
"""