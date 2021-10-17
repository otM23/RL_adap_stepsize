# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 16:58:48 2019

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
import constant as Estim_mean_bench
from errors import error_1
from plotting import Plot_plot
import numpy as np
import matplotlib.pyplot as plt


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

#### Benchmark
h_0 = np.array(h_0_init) # 100*np.random.rand(pdic['nb_iter'])#
pctg_0 = 0.01
h_01,h_01_past,error_h01 = Estim_mean_bench.Loop_super1(nb_episode,pdic,gamma= gamma,s_value=s_value,Error=Error,freq_print=freq_print,h_0=h_0,nb_iter = nb_iter,pctg_0 = pctg_0)
print(h_01.mean())

#### Plot improvement v1
start = 0
option_save = ""
path_Image = "Image"; ImageName = "\\improvement_drift_estim_bench_alone"
df = [ [error_h01[start:,0],np.log(error_h01[start:,1])]]
labels = ["step_cste"]
mark = ['o']
fig = plt.figure(figsize=(8,5))
Plot_plot(df,labels, xlabel ="Number of iteration", ylabel ="Log L2 - error",
               option=option_save, path =path_Image, ImageName=ImageName, Nset_tick_x = False, mark = mark)

