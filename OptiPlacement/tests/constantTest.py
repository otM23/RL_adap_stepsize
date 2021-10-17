# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 01:05:37 2019

@author: othmane.mounjid
"""

### Load paths 
import os 
import sys
Path_parent_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0,Path_parent_directory + "\\Plotting")
sys.path.insert(0,Path_parent_directory + "\\Utils")
sys.path.insert(0,Path_parent_directory + "\\OptiPlacement\\rLAlgorithms")

####### Import liraries 
import constant as op_place
import solTheo
from errors import error_1
import numpy as np 
import pandas as pd
import plotting as pltg
import matplotlib.pyplot as plt
import seaborn as sns

###############################################################################
###############################################################################
########################################## End functions ######################
###############################################################################
###############################################################################

#### Compute the result
######### Initialize the parameters ::Intensity values
path = Path_parent_directory + "\\OptiPlacement\\Data\\"
filename = "Intens_val_qr.csv"
Intens_val = pd.read_csv(path + filename, index_col = 0)
Intens_val_bis = Intens_val[Intens_val['Spread'] == 1].groupby(['BB size']).agg({'Limit':'mean', 'Cancel': 'mean', 'Market': 'mean'}).loc[:10,:]
Intens_val_bis.reset_index(inplace = True)
Intens_val_bis.loc[0,['Cancel','Market']] = 0
    
######### Show the database
print(Intens_val_bis.head(10))

######### Theoretical solution :  numerical scheme
tol = 0.1
size_q = Intens_val_bis.shape[0]
nb_iter_scheme = 400
reward_exec_1 = lambda qsame,bb_pos: op_place.reward_exec(qsame, bb_pos, gain = 6, cost_out = -0.6, cost_stay = -0.2)
df_bis = solTheo.Sol_num_scheme(nb_iter_scheme,size_q,Intens_val_bis,tol = tol,reward_exec_1 = reward_exec_1)


######### Simulation of the order book
##### Initialization of the parameters
np.random.seed(5)
qb_0 = 2# 4
bb_pos_0 = qb_0 -1
Lob_state_0 = [qb_0,bb_pos_0] 
write_option = True
nb_iter = 100
gamma = 0.1 ## learning rate
h_0_init = 5*np.ones((size_q,size_q+1))
h_0_Stay_init = np.ones((size_q,size_q+1))
h_0_Mkt_init = np.ones((size_q,size_q+1))
h_0 = np.array(h_0_init)
h_0_Stay = np.array(h_0_Stay_init)
h_0_Mkt = np.array(h_0_Mkt_init)
h_0_theo = solTheo.Read_h_0_theo(df_bis["Value_opt"].values,size_q,reward_exec_1)
Error = lambda x : error_1(np.nan_to_num(h_0_theo),np.nan_to_num(x))

######### Test RL methods 
########### global parameters
nb_episode = 300 ## upper loop iterations
freq_print = 40
nb_iter = 100 ## inner loop iterations
qb_0 = 2 # 4
bb_pos_0 = qb_0 -1
Lob_state_0 = [qb_0,bb_pos_0] ## initial state
eta = 1 ## discount factor
write_option = False

########### Benchmark 
h_0 = np.array(h_0_init)
h_0_Stay = np.array(h_0_Stay_init)
h_0_Mkt = np.array(h_0_Mkt_init)
pctg_0 = 0.05
h_01, h_01_past, error_h01, h_01_Stay, h_01_Mkt = op_place.Lob_simu_super_Loop1(nb_episode,Intens_val_bis,nb_iter, gamma= gamma,Lob_state_0=Lob_state_0,freq_print=freq_print,size_q = size_q,Error=Error, write_option = write_option, h_0 = h_0, reward_exec = reward_exec_1, eta = eta, h_0_Stay = h_0_Stay, h_0_Mkt = h_0_Mkt, pctg_0 = pctg_0)
print(error_h01)


###############################################################################
########################## 1- Plot the improvement ############################
###############################################################################

##### Plot v3
start = 0
end = 16
option_save = ""
path_Image = "Image"; ImageName = "\\improvement__optimal_placement_bench_"
df = [ [error_h01[start:end,0],np.log(error_h01[start:end,1])]]
labels = ["Benchmark "]
mark = ['o']
fig = plt.figure(figsize=(8,5))
pltg.Plot_plot(df,labels, xlabel ="Number of iterations", ylabel ="Log L2 - error",
               option=option_save, path =path_Image, ImageName=ImageName, Nset_tick_x = False, mark = mark)



###############################################################################
###############################################################################
########################## 2- Plot the values #################################
###############################################################################
###############################################################################

##### Plot the values
option=""; path = "Image"; ImageName="\\h___theo"; xtitle=""; figsize_=(8, 8)
a = 1; b = 1; subplot0 = 1; Nset_tick_x = False; annot =True; fmt = '.2g'; cbar = False; annot_size = 16
cmaps = [sns.diverging_palette(10,240,n=6)]; masks = [None]
tilde_size_q = size_q - 1
q_val_ = df_bis["Value_opt"]
x_val = np.arange(1,tilde_size_q+1)
y_val = np.arange(1,tilde_size_q+1)
nb_x = tilde_size_q
nb_y = tilde_size_q 
df_n_row = nb_x*nb_y
df = pd.DataFrame(np.zeros((df_n_row,3)),columns=['x','y','z'])
df['x'] = np.repeat(x_val,nb_y)
df['y'] = np.tile(y_val,nb_x)
df['z'] = q_val_#u_next_1
fig = plt.figure(figsize=figsize_)
pltg.Plot_sns_2(df.pivot("y", "x", "z"),option=option,path =path,ImageName=ImageName,xtitle=xtitle, xlabel = "", ylabel = "", annot = annot, fig = fig, a = a, b = b, subplot0 = subplot0, cbar = cbar, cmaps = cmaps, masks = masks, fmt = fmt, annot_size = annot_size )


##### Plot the result
option=""; path = "Image"; ImageName="\\h___01"; xtitle=""; figsize_=(8, 8)
a = 1; b = 1; subplot0 = 1; Nset_tick_x = False; annot =True; fmt = '.2g'; cbar = False; annot_size = 16
cmaps = [sns.diverging_palette(10,240,n=6)]; masks = [None]
q_val_ = np.around(h_01[1:,2:], decimals=1)
x_val = np.arange(1,size_q)
y_val = np.arange(1,size_q)
nb_x = size_q-1
nb_y = size_q-1
df_n_row = nb_x*nb_y
df = pd.DataFrame(np.zeros((df_n_row,3)),columns=['x','y','z'])
df['x'] = np.repeat(x_val,nb_y)
df['y'] = np.tile(y_val,nb_x)
df['z'] = q_val_.flatten()
fig = plt.figure(figsize=figsize_)
pltg.Plot_sns_2(df.pivot("y", "x", "z"),option=option,path =path,ImageName=ImageName,xtitle=xtitle, xlabel = "", ylabel = "", annot = annot, fig = fig, a = a, b = b, subplot0 = subplot0, cbar = cbar, cmaps = cmaps, masks = masks, fmt = fmt,annot_size=annot_size)


###############################################################################
###############################################################################
########################## 3- Plot the ctrl ###################################
###############################################################################
###############################################################################

##### Plot the values
option=""; path = "Image"; ImageName="\\ctrl_theo"; xtitle=""; figsize_=(8, 8)
a = 1; b = 1; subplot0 = 1; Nset_tick_x = False; annot =True; fmt = '.2g'; cbar = False; annot_size = 16
cmaps = [sns.diverging_palette(10,240,n=6)]; masks = [None]
tilde_size_q = size_q - 1
q_val_ = (df_bis["Limit"]>=df_bis["Market"]).astype(int)
for qsame in range(size_q-1): # qsame is the size
    q_val_[qsame*(size_q -1) + qsame +1: (qsame+1)*(size_q -1)] = np.nan
x_val = np.arange(1,tilde_size_q+1)
y_val = np.arange(1,tilde_size_q+1)
nb_x = tilde_size_q
nb_y = tilde_size_q 
df_n_row = nb_x*nb_y
df = pd.DataFrame(np.zeros((df_n_row,3)),columns=['x','y','z'])
df['x'] = np.repeat(x_val,nb_y)
df['y'] = np.tile(y_val,nb_x)
df['z'] = q_val_ #u_next_1
fig = plt.figure(figsize=figsize_)
pltg.Plot_sns_2(df.pivot("y", "x", "z"),option=option,path =path,ImageName=ImageName,xtitle=xtitle, xlabel = "", ylabel = "", annot = annot, fig = fig, a = a, b = b, subplot0 = subplot0, cbar = cbar, cmaps = cmaps, masks = masks, fmt = fmt, annot_size = annot_size )


##### Plot the result
option=""; path = "Image"; ImageName="\\ctrl__h__01"; xtitle=""; figsize_=(8, 8)
a = 1; b = 1; subplot0 = 1; Nset_tick_x = False; annot =True; fmt = '.2g'; cbar = False; annot_size = 16
cmaps = [sns.diverging_palette(10,240,n=6)]; masks = [None]
q_val_ = (h_01_Stay >= h_01_Mkt)[1:,2:].astype(float)
for qsame in range(size_q-1): # qsame is the size
    q_val_[qsame,(qsame+1):] = np.nan
x_val = np.arange(1,size_q)
y_val = np.arange(1,size_q)
nb_x = size_q-1
nb_y = size_q-1
df_n_row = nb_x*nb_y
df = pd.DataFrame(np.zeros((df_n_row,3)),columns=['x','y','z'])
df['x'] = np.repeat(x_val,nb_y)
df['y'] = np.tile(y_val,nb_x)
df['z'] = q_val_.flatten()
fig = plt.figure(figsize=figsize_)
pltg.Plot_sns_2(df.pivot("y", "x", "z"),option=option,path =path,ImageName=ImageName,xtitle=xtitle, xlabel = "", ylabel = "", annot = annot, fig = fig, a = a, b = b, subplot0 = subplot0, cbar = cbar, cmaps = cmaps, masks = masks, fmt = fmt,annot_size=annot_size)