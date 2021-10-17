# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 08:46:43 2019

@author: othmane.mounjid
"""

####### Load paths 
import os 
import sys
Path_parent_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0,Path_parent_directory + "\\Plotting")
sys.path.insert(0,Path_parent_directory + "\\Utils")

####### Import liraries 
import numpy as np 
import errors as proj_op


###############################################################################
###############################################################################
########################################## Start RL functions #################
###############################################################################
###############################################################################

#### Inner loops  

###### Alg 2 inner loop : SAGA version test 2: methodology of visiting states of the proximal gradient algorithm
def Loop_within_episode_5_2(v_0,v_0_past,nb_past,exp_mean,s_init,x_init,q_init,pdic,gamma =0.01,Error=None, n_max =  2, prox_op = proj_op.prox_id):
    ####### Initial condition
    v_0_cum = np.zeros(((pdic['nb_iter']+1),pdic['size_q'])) ## update function
    s_value = float(s_init) ## initial price value
    x_value = float(x_init) ## initial wealth
    q_value = float(q_init) ## initial inventory
    ###### Random values
    rnd_values = np.sqrt(pdic['Time_step']*pdic['sigma2'])*np.random.normal(loc=0.0, scale=1.0, size=pdic['nb_iter']) ## generates random gaussian variables
    nu_value = 0 ## initial control is zero # pdic['Step_q']/pdic['Time_step'] # agent_decision(0,s_value,x_value,q_value,h_0,h_1,h_2,pdic)
    q_indeces = np.arange(pdic['size_q'])
    
    ###### Forward loop
    for i in range(pdic['nb_iter']): # i = 0
        s_value_next = s_value + pdic['alpha']*pdic['mu']*pdic['Time_step'] + rnd_values[i]
        x_value_next = x_value - pdic['kappa']*(nu_value*nu_value)*pdic['Time_step'] - nu_value*s_value*pdic['Time_step']
        q_value_next = min(max(q_value + nu_value*pdic['Time_step'],pdic['Q_min']),pdic['Q_max'])
        
        
        ### Update v_0 and v_0_cum
        ##### Next controls : quantities to consume
        q_val_min = max(pdic['Q_min'],pdic['Q_min']-q_value) ## we put pdic['Q_min'] for safety 
        q_val_max = min(pdic['Q_max'],pdic['Q_max']-q_value)
        q_consump_values = np.arange(q_val_min,q_val_max,pdic['Step_q'])
        q_consump_values = q_consump_values[q_consump_values != 0]

        q_next_values = (q_value + q_consump_values) ## next inventories
        nu_next_values = q_consump_values/pdic['Time_step'] ## next trading speeds
        iq_next_values = np.rint(np.minimum(np.maximum((q_next_values - pdic['Q_min'])/pdic['Step_q'],0),pdic['size_q']-1)).astype(int)     
        indexq_q = int(round(min(max((q_value - pdic['Q_min'])/pdic['Step_q'],0),pdic['size_q']-1)))
        ## next wealths
        x_values_next = x_value - pdic['kappa']*(nu_next_values*nu_next_values)*pdic['Time_step'] - nu_next_values * s_value * pdic['Time_step']
        
        ## next rewards
        vect_values = (x_values_next - x_value) + (q_next_values*s_value_next - q_value*s_value) - pdic['phi'] * q_value * q_value * pdic['Time_step'] +  v_0[i+1,iq_next_values] - v_0[i,indexq_q]
        v_0_cum[i,indexq_q] += vect_values.max() ## update v_0_cum   
            
            
        ## Update of the values and stored values
        j = nb_past[i,indexq_q] # np.random.randint(self._n_max)        
        if j == 0:
            v_0[i,indexq_q] += gamma*(v_0_cum[i,indexq_q])
            v_0_past[i,indexq_q,nb_past[i,indexq_q]] = v_0_cum[i,indexq_q]             
        elif j < n_max -1:
            r = np.random.randint(0,j)
            nb_values = exp_mean[:j].sum()#exp_mean.sum() #j + 1
            v_0[i,indexq_q] = v_0[i,indexq_q] +  gamma*(v_0_cum[i,indexq_q] - v_0_past[i,indexq_q,r] + (((v_0_past[i,indexq_q,:j]*exp_mean[:j]).sum())/(nb_values)))
            v_0_past[i,indexq_q,nb_past[i,indexq_q]] = v_0_cum[i,indexq_q]  
        else:
            r = np.random.randint(0,n_max)
            nb_values = exp_mean[:].sum()#exp_mean.sum() #j + 1
            v_0[i,indexq_q] = v_0[i,indexq_q] +  gamma*(v_0_cum[i,indexq_q] - v_0_past[i,indexq_q,r] + (((v_0_past[i,indexq_q,:]*exp_mean[:]).sum())/(nb_values)))
            v_0_past[i,indexq_q,r] = v_0_cum[i,indexq_q] 

        nb_past[i,indexq_q] = min(nb_past[i,indexq_q]+1,n_max-1)
        

        ### Choose next  values       
        s_value = s_value_next
        x_value = x_value_next
        q_value = q_value_next
        q_aux = np.exp(5*np.abs(v_0_past[i+1,q_indeces,np.maximum(nb_past[i+1,:]-1,0)])) + 1e-4
        i_q_next_aux = np.random.choice(pdic['size_q'], 1, p= q_aux/q_aux.sum())[0]
        nu_value = (pdic['Q_min'] + i_q_next_aux * pdic['Step_q'] - q_value)/pdic['Time_step']
     
    error_val = Error(v_0)
    return [v_0, error_val,v_0_past,nb_past]


#### Super loops
     
###### Alg 1 super loop : average value over the past observations :: this function takes inner loops functions as an argument :: allowed inner loops functions are : Loop_within_episode_4_1 -- Loop_within_episode_4_2 -- Loop_within_episode_5_1 -- Loop_within_episode_5_2 
def Loop_all_episode_2(nb_episode,pdic,inner_loop_func = Loop_within_episode_5_2,gamma= 0.2,freq_print=100,Error=None, v_0 = None, print_option =  True, n_max =  2, _exp_mean = 1, v_0_past = None, nb_past = None, pctg_0 = 0.01):
    size_mean = int(nb_episode//freq_print)
    if (v_0 is None) :
        v_0 = 1*np.ones(((pdic['nb_iter']+1),pdic['size_q']))
    if (v_0_past is None) :
        v_0_past = np.zeros(((pdic['nb_iter']+1),pdic['size_q'],n_max))
    if (nb_past is None) :
        nb_past = np.zeros(((pdic['nb_iter']+1),pdic['size_q']),dtype = int)
    q_values = np.arange(-pdic['Q_max'],pdic['Q_max'],pdic['Step_q'])
    q_indeces = np.arange(pdic['size_q'])
    v_0[-1,:] = -pdic['A']*q_values*q_values
    

#    v_0_past = np.zeros(((pdic['nb_iter']+1),pdic['size_q'],n_max))
#    nb_past = np.zeros(((pdic['nb_iter']+1),pdic['size_q']),dtype = int)
    exp_mean = _exp_mean**(np.arange(n_max,0,-1))  
    
    error_within = np.zeros(freq_print)
    count_within = 0
    count_reward = 0
    mean_reward = np.zeros((size_mean,2))
    count_period = 0
    gamma_0 = float(gamma)
    ## upper level : copy of old values
    v_0_before = np.array(v_0)
    v_0_past_before = np.array(v_0_past)
    nb_past_before = np.array(nb_past) 
    for ep in range(nb_episode): # ep = 0
        s_init = 0 # proj_op.choose_elt_rnd(pdic['size_s'],pdic['s_min'],pdic['Step_s']) # 0#5 #choose_elt(pdic['size_s']+1,pdic['s_min'],pdic['Step_s'],{'index':5})
        x_init = 1 # proj_op.choose_elt_rnd(pdic['size_x'],pdic['x_min'],pdic['Step_x']) # 0 #choose_elt(pdic['size_x']+1,pdic['x_min'],pdic['Step_x'],{'index':5})
        q_aux =  np.abs(v_0_past[0][q_indeces,np.maximum(nb_past[0,:]-1,0)])
        q_aux[q_aux == 0] =1
        q_aux = np.exp(5*q_aux) + 1e-4
        i_q_next_aux = np.random.choice(pdic['size_q'], 1, p= q_aux/q_aux.sum())[0]
        q_init = pdic['Q_min'] + i_q_next_aux * pdic['Step_q']

        v_0,error_val,v_0_past,nb_past = inner_loop_func(v_0,v_0_past,nb_past,exp_mean,s_init,x_init,q_init,pdic,gamma = gamma_0, n_max = n_max, Error = Error)

        error_within[count_within] = error_val
        count_within += 1
        if (((ep %(freq_print))==freq_print-1) and (ep> 0)):
            if print_option:
                print(" frequency is : " + str(ep))
            mean_reward[count_reward] = (count_reward*freq_print,error_within.mean())
            ## update gamma :
            index_count_before = max(count_reward-1,0)
            pctg_last = ((mean_reward[index_count_before,1] - mean_reward[count_reward,1])/mean_reward[index_count_before,1])
            if (pctg_last <= pctg_0) and (count_reward>= 1):
                v_0 = np.array(v_0_before)
                v_0_past = np.array(v_0_past_before)
                nb_past = np.array(nb_past_before) 
                if (count_period >= 5):
                    gamma_0 = (gamma_0 - 0.01)
                    count_period = 0
                print(gamma_0)
                print(pctg_last)
                print(count_period)
                count_period += 1
            else : 
                v_0_before = np.array(v_0)
                v_0_past_before = np.array(v_0_past)
                nb_past_before = np.array(nb_past) 
            error_within[:] = 0
            count_within = 0
            count_reward += 1
    if (count_within==0):
        return  [v_0,mean_reward,gamma_0]
    else:
        val = np.array((count_reward*freq_print,error_within[:count_within].mean())).reshape((-1,2))
        return [v_0,np.concatenate((mean_reward,val)),gamma_0] 