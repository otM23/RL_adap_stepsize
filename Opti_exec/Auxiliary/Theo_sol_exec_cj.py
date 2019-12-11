# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 13:54:13 2019

@author: othmane.mounjid
"""

## Import libraries
import numpy as np


def Compute_h_2(pdic):
    a_1 = np.sqrt(4 * pdic['kappa'] * pdic['phi'])
    a_2 = (-1/(pdic['kappa'])) * a_1
    a_3 = -1/(2 * a_2 * pdic['kappa'])
    a_4 = 1/(2*pdic['A'] + a_1) - a_3
    time_values = np.arange(0,pdic['T_max']+pdic['Time_step'],pdic['Time_step'])
    h_2 = 1/(a_3 + a_4*np.exp(a_2 * (pdic['T_max'] - time_values))) - a_1
    return h_2


def Compute_h_1(nb_iter, h_2, pdic):
    h_1 = np.zeros((nb_iter+1))
    
    ###### Terminal condition
    h_1[-1] = 0
    
    ###### Backward loop 
    for i in range(nb_iter): # i = 0
        h_1[nb_iter-(i+1)] = h_1[nb_iter-i] - (((h_2[nb_iter-i])/(2*pdic['kappa'])) * h_1[nb_iter-i] - pdic['alpha']*pdic['mu'] ) * pdic['Time_step']

    return h_1

def Compute_h_0(nb_iter,h_1, pdic):
    h_0 = np.zeros((nb_iter+1))
    
    ###### Terminal condition
    h_0[-1] = 0
    
    ###### Backward loop 
    for i in range(nb_iter): # i = 0
        h_0[nb_iter-(i+1)] = h_0[nb_iter-i] - (-(h_1[nb_iter-i] * h_1[nb_iter-i])/( 4 * pdic['kappa'])) * pdic['Time_step']

    return h_0