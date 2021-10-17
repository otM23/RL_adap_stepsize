# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 16:51:37 2019

@author: othmane.mounjid
"""

####### Import liraries 
import numpy as np 
import matplotlib.pyplot as plt

####### RL approach

#### Step 1 : estimation of the error: auxiliary function
def error_1(bench,x):
    return np.linalg.norm(x -bench)

#### Step 2 : plot function : auxiliary function 
def Plot_plot(df,labels,option=False,path ="",ImageName="",xtitle="", xlabel ="", ylabel ="", fig = False, a = 0, b = 0, subplot0 = 0, linewidth= 3.0, Nset_tick_x = True, xlim_val = None, ylim_val = None,mark = None, col = 'blue',marksize=12, bbox_to_anchor_0 = (0.7,.95)):
    if mark is None:
        mark = ['o']*(len(df))
    if not fig:
        ax = plt.axes()
    else:
        ax = fig.add_subplot(a,b,subplot0)
    
    count = 0
    for elt in df:
        ax.plot(elt[0],elt[1], label = labels[count], linewidth= linewidth, marker = mark[count],  markersize = marksize)
        count +=1
    ax.set_title(xtitle,fontsize = 18)
    ax.set_xlabel(xlabel,fontsize = 18)
    ax.set_ylabel(ylabel,fontsize = 18)
    ax.grid(b=True)
    ax.legend(loc = 2, bbox_to_anchor = bbox_to_anchor_0)
    if Nset_tick_x:
        ax.set_xticklabels([])
    if xlim_val :
        ax.set_xlim(xlim_val)
    if ylim_val :
        ax.set_ylim(ylim_val)
#    ax.get_xaxis().set_visible(False)
#    ax.get_yaxis().set_visible(False)
    if option == "save" :
        plt.savefig(path+ImageName+".pdf", bbox_inches='tight') 
        
###############################################################################
###############################################################################
########################################## Start functions ####################
###############################################################################
###############################################################################

#### Inner loop
###### Benchmark inner loop
def Loop1(h_0,h_0_past,pdic,gamma= 0.2,s_value=0,Error=None, nb_iter = 100):
    s_value_next = 0
    rnd_values = np.sqrt(pdic['Time_step']*pdic['sigma2'])*np.random.normal(loc=0.0, scale=1.0, size=pdic['nb_iter'])
    h_0_cum = np.zeros((pdic['nb_iter']+1))
    ###### Forward loop
    for i in range(nb_iter): # i = 0
        s_value_next = s_value + pdic['alpha']*pdic['mu']*pdic['Time_step'] + (rnd_values[i])
        h_0_cum[i] += (s_value_next - s_value - h_0[i])
        
        ### Update the value
        h_0[i] = h_0[i] + gamma*h_0_cum[i]

        h_0_past[i] = h_0_cum[i] 
        
        s_value = s_value_next
        
    error_val = Error(h_0)
    return [h_0,h_0_past,error_val]

#### Super loops
###### Benchmark super loop
def Loop_super1(nb_episode,pdic,gamma= 0.2,s_value=0,freq_print=100,Error=None, h_0 = None, print_option =  True,nb_iter = 100, pctg_0 = 0.1):
    size_mean = int(nb_episode//freq_print)
    if (h_0 is None) :
        h_0 = 100*np.ones((pdic['nb_iter']))
    h_0_past = np.zeros((pdic['nb_iter']+1))
    error_within = np.zeros(freq_print)
    error_within_estim = np.zeros(freq_print)
    count_within = 0
    count_reward = 0
    mean_reward = np.zeros((size_mean,2))
    mean_error_estim = np.zeros((size_mean,2))
    gamma_0 = float(gamma)
    ### Parameters ; upper level 
    h_0_before = np.array(h_0)
    h_0_past_before = np.array(h_0_past)
    count_period =0
    for ep in range(nb_episode):
        h_0,h_0_past,error_val = Loop1(h_0,h_0_past,pdic,gamma= gamma_0,s_value=s_value,Error=Error,nb_iter = nb_iter)
        error_within[count_within] = error_val
        error_within_estim[count_within] = np.linalg.norm(h_0_past)
        count_within += 1
        if (((ep %(freq_print))==freq_print-1) and (ep> 0)):
            if print_option:
                print(" frequency is : " + str(ep))
            mean_reward[count_reward] = (count_reward*freq_print,error_within.mean())
            mean_error_estim[count_reward] = (count_reward*freq_print,error_within_estim.mean())
            ## update gamma :
            index_count_before = max(count_reward-1,0)
            pctg_last = ((mean_error_estim[index_count_before,1] - mean_error_estim[count_reward,1])/mean_error_estim[index_count_before,1])
            if (pctg_last <= pctg_0) and (count_reward >= 1):
                h_0 = np.array(h_0_before)
                h_0_past = np.array(h_0_past_before)                
                if (count_period >= 3):
                    gamma_0 = max(gamma_0/2,0.01) # count_gamma_0 = 1
                    count_period = 0
            else : 
                h_0_before = np.array(h_0)
                h_0_past_before = np.array(h_0_past)               
            error_within[:] = 0
            count_within = 0
            count_reward += 1
            count_period += 1
    if (count_within==0):
        return  [h_0,h_0_past,mean_reward]
    else:
        val = np.array((count_reward*freq_print,error_within[:count_within].mean())).reshape((-1,2))
        return [h_0,h_0_past,np.concatenate((mean_reward,val))]      
