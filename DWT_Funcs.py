# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 13:07:48 2023

Description:
    A prog that contains several functions used in Density_Wave_Theory_Main.py 
    so as to decluster the main prog slightly.


@author: James Acton (20325303)
"""

#importing libraries
import numpy as np
import scipy.stats as stats
import random



#Galaxy defining function
class Galaxy:
    """
    

    Parameters
    ----------
    N : int
        Number of stars.
    r_core : float
        galaxy core radius.
    r_galax : float
        galaxy radius.
    e1 : flaot
        inner ellipticity.
    e2 : float
        outer ellipticity.
    ang_offset : TYPE
        angular offset.
    vel : float
        velocity.
    t_step : float
        time step.

    Returns
    -------
    None.

    """
    def __init__(self, N, r_core, r_galax, e1, e2, ang_offset, vel, t_step):

        self.N = N
        self.r_core = r_core 
        self.r_galax = r_galax 
        self.r_dist = 2*r_galax 
        self.e1 = e1
        self.e2 = e2
        self.ang_offset = ang_offset
        self.vel = vel
        self.t_step = t_step
 

#function that generates/calculates the semi-amjor axis, initial angle values,
# ellipticity, semi-minor axis, and initial x and y coords for a galaxy.
def calculations(N, r_core, r_galax, e1, e2, ang_offset):
    """
    

    Parameters
    ----------
    N : int
        Number of stars.
    r_core : float
        galaxy core radius.
    r_galax : float
        Galaxy radius.
    e1 : float
        inner ellipticity.
    e2 : float
        outer ellipticity.
    ang_offset : float
        angular offset.

    Returns
    -------
    x_vals : array
        array of x-coords for plotting.
    y_vals : array
        array of y-coords for plotting.
    ang_vals : array
        array of angle values.
    a_vals : array
        arrary of semi-major axis values.
    orb_tilts : array
        array of orbital tilts.
    b_vals : array 
        array of semi-minor axis values.

    """
    #Setting values for normal distribution
    low, upp = -r_galax, r_galax
    a_mean = 0.0
    sd = r_galax/3
    
    #-generating random semi-major axis and angle values-#
    a_vals = []
    ang_vals = []
    
    for i in range(N):
        a_vals.append(stats.truncnorm.rvs(                                   #truncated normal distribution centred at 0.0, stand dev of r_galax/3,
            (low - a_mean) / sd, (upp - a_mean) / sd, loc=a_mean, scale=sd)) #bound between +/- r_galax
        rand_ang = random.uniform(0, 2*np.pi)
        ang_vals.append(rand_ang) #generating an list of random angle values ranging from 0 to 2pi
    a_vals = np.array(a_vals) #converting list to array
    ang_vals = np.array(ang_vals) #converting list to array


    #-calculating ellipticity-#
    E_vals = []

    for a in a_vals:
        if a <= r_core and a >= -r_core:
            E = 1 + (a/r_core)*(e1 - 1)
        else:
            E = e1 + ((a - r_core)/(r_galax - r_core))*(e2 - e1)
        E_vals.append(E)

    E_vals = np.array(E_vals) #converting list to array

    b_vals = E_vals*a_vals #semi-minor axes

    orb_tilts = -np.pi/2 + (a_vals)*ang_offset #generating array of orbital tilt values

    x_vals = (a_vals*np.cos(ang_vals)*np.cos(orb_tilts) 
              - b_vals*np.sin(ang_vals)*np.sin(orb_tilts)) #generating x coords for each star

    y_vals = (a_vals*np.cos(ang_vals)*np.sin(orb_tilts) 
              + b_vals*np.sin(ang_vals)*np.cos(orb_tilts)) #generating y coords for each star
    
    return x_vals, y_vals, ang_vals, a_vals, orb_tilts, b_vals


#function that generates arrays of x and y coordinates for animating a galaxy,
# given the initial x and y coords as well as the velocity at which the stars
# are moving.
def anim_vals(x, y, angles, frms, vel, t_step, a, tilts, b):
    """
    

    Parameters
    ----------
    x : array
        array of x value arrays.
    y : array
        array of y value arrays.
    angles : array
        array of angle value arrays.
    frms : int
        number of frames to be animated.
    vel : float
        angular velocity for change in position.
    t_step : float
        Time step for change in position.
    a : array
        Semi-major axis values.
    tilts : array
        orbital tilt values.
    b : array
        Semi-minor axis values.

    Returns
    -------
    an_x_vals : array
        array of all x value arrays for animating.
    an_y_vals : array
        array of all y value arrays for animating.

    """
    #creating lists of zeros of length = to the amount of frames for animating
    an_x_vals = [0]*frms
    an_y_vals = [0]*frms
    an_ang_vals = [0]*frms

    #setting the first value of each list equal to the initial values
    an_x_vals[0] = x
    an_y_vals[0] = y
    an_ang_vals[0] = angles

    for i in range(frms-1):
        an_ang_vals[i+1] = (an_ang_vals[i] + vel*t_step)
        an_x_vals[i] = (a*np.cos(an_ang_vals[i])*np.cos(tilts) 
                        - b*np.sin(an_ang_vals[i])*np.sin(tilts))
        
        an_y_vals[i] = (a*np.cos(an_ang_vals[i])*np.sin(tilts) 
                        + b*np.sin(an_ang_vals[i])*np.cos(tilts))
    
    return an_x_vals, an_y_vals


#function that generates a brightness distribution array for a given galaxy.   
def bright_distrib(x, y, r_core, r_galax):
    """
    

    Parameters
    ----------
    x : array
        array of x values.
    y : array
        array of y values.
    r_core : float
        radius of galactic core.
    r_galax : float
        radius of galaxy.

    Returns
    -------
    I : array
        array of intensity values.

    """
    r = 4e-8*(np.sqrt((x**2)+(y**2))) #scaled radial distance
    I_0 = 1e9 # central intensity
    k = 7.669 # constant = Beta*ln10, where Beta is another coefficient
    R_D = 4e-8*(r_core/r_galax) #ratio between r_core and r_galax
    I = np.array([0]*len(x)) #creating an array of zeros equal to the length of the x values
    for i in range(len(x)):      
        if r[i] <= r_core:
            I[i] = I_0*np.exp(-k*(r[i]**(1/4)))
        elif r[i] > r_core and r[i] <= r_galax:
            I[i] = I_0*np.exp(-r[i]/R_D)
    return I



#function to generate initial array of positions for H-II regions in galaxy
def H_II(N, u1, v1, a1, b1, u2, v2, a2, b2, st, mid, end, frms):
    """

    Parameters
    ----------
    N : int
        Number of H-II regions on given elliptical path.
    u1 : int
        x-coord of centre of ellipse 1.
    v1 : int
        y-coord of centre of ellipse 1.
    a1 : int
        Semi-major radius of ellipse 1.
    b1 : int
        Semi-minor radius of ellipse 1.
    u2 : int
        x-coord of centre of ellipse 2.
    v2 : int
        y-coord of centre of ellipse 2.
    a2 : int
        Semi-major radius of ellipse 2.
    b2 : int
        Semi-minor radius of ellipse 2.
    st : float
        starting angle of ellipse section 1.
    mid : float
        end angle of ellipse section 1, and start angle of ellipse section 2.
    end : float
        end angle of ellipse section 2.
    frms : int
        number of frames to be animated.

    Returns
    -------
    an_t1 : list
        list of arrays for animating H-II regions along ellipse section 1.
    an_t2 : list
        list of arrays for animating H-II regions along ellipse section 2.

    """
    
    #assigning random initial positions along ellipse sections
    t1s = []
    t2s = []
    for i in range(N):    
        t1 = random.uniform(st, mid)
        t1s.append(t1)
        
        t2 = random.uniform(mid, end)
        t2s.append(t2)
        
    t1s = np.array(t1s)
    t2s = np.array(t2s)


    an_t1 = [0]*frms
    an_t2 = [0]*frms

    an_t1[0] = t1s
    an_t2[0] = t2s
    for i in range(frms-1):
        an_t1[i+1] = an_t1[i] - np.pi/100
        an_t2[i+1] = an_t2[i] - np.pi/100
        
    #two loops below keep point contained within certain sections of the ellipses
    for j in range(len(an_t1)):
        for k in range(len(an_t1[j])):
            if an_t1[j][k] < st:
                an_t1[j][k] = random.uniform(st, mid)
                    
    for j in range(len(an_t2)):
        for k in range(len(an_t2[j])):
            if an_t2[j][k] < mid:
                an_t2[j][k] = random.uniform(mid, end)
    
    return an_t1, an_t2
