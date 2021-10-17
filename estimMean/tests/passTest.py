# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 20:33:20 2019

@author: othmane.mounjid
"""

### Load paths 
import os 
import sys
Path_parent_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0,Path_parent_directory + "\\Plotting")
sys.path.insert(0,Path_parent_directory + "\\Utils")
sys.path.insert(0,Path_parent_directory + "\\estimMean\\rLAlgorithms")

### Import libraries
import pass_ as Estim_mean_pass
from errors import error_1
from plotting import Plot_plot
import numpy as np
import matplotlib.pyplot as plt

### Load paths 
import os 
import sys
Path_parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,Path_parent_directory + "\\Plotting")

###############################################################################
###############################################################################
########################################## End functions ######################
###############################################################################
###############################################################################

    
#### Compute the result
######### Initialize the parameters 
gamma = 0.1 ## learning rate
s_value = 0 ## price init value
T_max = 1
nb_iter = 100
Time_step = (T_max)/nb_iter
alpha = 0.1
mu = 1
sigma2 = 5 # 5 # 1
pdic = {  'Time_step':Time_step,
          'alpha':alpha,
          'mu':mu,
          'T_max':T_max,
          'nb_iter':nb_iter,
          'sigma2':sigma2}

### Initialization rl algorithm
h_0_init = 10*np.ones((pdic['nb_iter'])) # 100*np.random.rand(pdic['nb_iter'])#
nb_episode = 100
freq_print = 5
h_0theo = pdic['alpha']*pdic['mu']*pdic['Time_step']*np.ones(nb_iter) ## theoretical value
Error = lambda x : error_1(h_0theo,x)


#### Alg pass : v_2 : increase gamma same sign **
_exp_mean = 1 ## decrease of the kernel :  when 1 it is the empirical average
n_max = 9 # 7 ## nb past values to store
h_0 = np.array(h_0_init)
gamma= 0.1
alpha_0 = 1
alpha_max = 3
pctg_0 = 0.01
alpha_init = 1
h_012,h_0_past12,nb_past12,error_h012,gamma_012 = Estim_mean_pass.Loop_super5(nb_episode,_exp_mean,pdic,inner_loop_func = Estim_mean_pass.Loop10, n_max=n_max,gamma= gamma,s_value=s_value,Error = Error,freq_print=freq_print,h_0=h_0,alpha_0 = alpha_0, alpha_max = alpha_max, pctg_0 = pctg_0, alpha_init = alpha_init)
print(h_012.mean())
print(gamma_012)


## Plot improvement v1
start = 0
option_save = ""
path_Image = "Image"; ImageName = "\\improvement_drift_estim"
df = [ [error_h012[start:,0],np.log(error_h012[start:,1])]]
labels = ["PASS"]
mark = ['o']
fig = plt.figure(figsize=(8,5))
Plot_plot(df,labels, xlabel ="Number of iteration", ylabel ="Log L2 - error",
               option=option_save, path =path_Image, ImageName=ImageName, Nset_tick_x = False, mark = mark)