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

###### Alg 2 inner loop v2 : increase gamma same sign : of the proximal gradient algorithm 
def Loop_within_episode_7(v_0,v_0_past,nb_past,s_init,x_init,q_init,pdic,gamma =0.01,Error=None, alpha_0 = 1, alpha_base = 1, alpha_max = 10, r = 0.5):
    ####### Initial condition
    v_0_cum = np.zeros(((pdic['nb_iter']+1),pdic['size_q'])) ## update function
    s_value = float(s_init) ## initial price value
    x_value = float(x_init) ## initial wealth
    q_value = float(q_init) ## initial inventory
    ###### Random values
    rnd_values = np.sqrt(pdic['Time_step']*pdic['sigma2'])*np.random.normal(loc=0.0, scale=1.0, size=pdic['nb_iter']) ## generates random gaussian variables
    nu_value = 0 ## initial control is zero # pdic['Step_q']/pdic['Time_step'] # agent_decision(0,s_value,x_value,q_value,h_0,h_1,h_2,pdic)
    alpha = alpha_0
    q_indeces = np.arange(pdic['size_q'])
    
    ###### Forward loop
    for i in range(pdic['nb_iter']): # i = 10
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
        indexq_q = int(round(min(max((q_value - pdic['Q_min'])/pdic['Step_q'],0),pdic['size_q']-1))) # q_value = 1.499999999900000003
        ## next wealths
        x_values_next = x_value - pdic['kappa']*(nu_next_values*nu_next_values)*pdic['Time_step'] - nu_next_values * s_value * pdic['Time_step']
        
        ## next rewards
        vect_values = (x_values_next - x_value) + (q_next_values*s_value_next - q_value*s_value) - pdic['phi'] * q_value * q_value * pdic['Time_step'] +  v_0[i+1,iq_next_values] - v_0[i,indexq_q]
        v_0_cum[i,indexq_q] += vect_values.max() ## update v_0_cum


        ## Update of the values and stored values 
        j = nb_past[i,indexq_q] # np.random.randint(self._n_max)  
        if j == 0:
            ## Update of the values and stored values
            gamma_move = alpha
            v_0[i,indexq_q] = v_0[i,indexq_q] + gamma*gamma_move*v_0_cum[i,indexq_q]
        elif (v_0_cum[i,indexq_q]*v_0_past[i,indexq_q]>=0):
            alpha = min(alpha +1,alpha_max)
            gamma_move = (1 + r * (alpha - 1))
            ## Update of the values and stored values 
            v_0[i,indexq_q] = v_0[i,indexq_q] + gamma*gamma_move*v_0_cum[i,indexq_q]
        elif (v_0_cum[i,indexq_q]*v_0_past[i,indexq_q] < 0):
            ### reinitialization
            alpha =  max(alpha -1,alpha_base)
            gamma_move = (1 + r * (alpha - 1))
            ## Update of the values and stored values 
            v_0[i,indexq_q] = v_0[i,indexq_q] + gamma*gamma_move*v_0_cum[i,indexq_q]
            

        v_0_past[i,indexq_q] = v_0_cum[i,indexq_q] 
        nb_past[i,indexq_q] = 1
        

        ### Choose next  values       
        s_value = s_value_next
        x_value = x_value_next
        q_value = q_value_next
        q_aux = np.exp(5*np.abs(v_0_past[i+1,q_indeces])) + 1e-4
        i_q_next_aux = np.random.choice(pdic['size_q'], 1, p= q_aux/q_aux.sum())[0]
        nu_value = (pdic['Q_min'] + i_q_next_aux * pdic['Step_q'] - q_value)/pdic['Time_step']
     
    error_val = Error(v_0)
    return [v_0, error_val,alpha,v_0_past,nb_past]

#### Super loops

###### Alg 3 super loop : average value over the past observations :: this function takes inner loops functions as an argument :: allowed inner loops functions are : Loop_within_episode_4_1 -- Loop_within_episode_4_2 -- Loop_within_episode_5_1 -- Loop_within_episode_5_2 
def Loop_all_episode_4(nb_episode,pdic,inner_loop_func = Loop_within_episode_7,gamma= 0.2,freq_print=100,Error=None, v_0 = None, print_option =  True, alpha_0 = 1, alpha_max = 10, v_0_past = None, nb_past = None, pctg_0 = 0.01, r = 0.5, alpha_base = 1):
    size_mean = int(nb_episode//freq_print)
    if (v_0 is None) :
        v_0 = 1*np.ones(((pdic['nb_iter']+1),pdic['size_q']))
    if (v_0_past is None) :
        v_0_past = np.zeros(((pdic['nb_iter']+1),pdic['size_q']))
    if (nb_past is None) :
        nb_past = np.zeros(((pdic['nb_iter']+1),pdic['size_q']),dtype = int)
    q_values = np.arange(-pdic['Q_max'],pdic['Q_max'],pdic['Step_q'])
    q_indeces = np.arange(pdic['size_q'])
    v_0[-1,:] = -pdic['A']*q_values*q_values
    
    error_within = np.zeros(freq_print)
    count_within = 0
    count_reward = 0
    mean_reward = np.zeros((size_mean,2))
    alpha = int(alpha_0)
    r_0 = float(r)
    count_period = 0
    gamma_0 = float(gamma)
    ### Upper level : copy past values
    v_0_before = np.array(v_0)
    v_0_past_before = np.array(v_0_past)
    nb_past_before = np.array(nb_past) 
    for ep in range(nb_episode): # ep = 0
        s_init = 5 # proj_op.choose_elt_rnd(pdic['size_s'],pdic['s_min'],pdic['Step_s']) # 0#5 #choose_elt(pdic['size_s']+1,pdic['s_min'],pdic['Step_s'],{'index':5})
        x_init = 0 # proj_op.choose_elt_rnd(pdic['size_x'],pdic['x_min'],pdic['Step_x']) # 0 #choose_elt(pdic['size_x']+1,pdic['x_min'],pdic['Step_x'],{'index':5})
        q_aux =  np.abs(v_0_past[0,q_indeces])
        q_aux[q_aux == 0] =1
        q_aux = np.exp(5*q_aux) + 1e-4
        i_q_next_aux = np.random.choice(pdic['size_q'], 1, p= q_aux/q_aux.sum())[0]
        q_init = pdic['Q_min'] + i_q_next_aux * pdic['Step_q']

        v_0,error_val,alpha,v_0_past,nb_past = inner_loop_func(v_0,v_0_past,nb_past,s_init,x_init,q_init,pdic,gamma = gamma_0, Error = Error, alpha_0 = alpha, alpha_max = alpha_max, r = r_0, alpha_base = alpha_base)

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
                alpha = int(alpha_0) 
                count_period += 1
            else : 
                v_0_before = np.array(v_0)
                v_0_past_before = np.array(v_0_past)
                nb_past_before = np.array(nb_past)       
            error_within[:] = 0
            count_within = 0
            count_reward += 1
    if (count_within==0):
        return  [v_0,mean_reward,v_0_past,nb_past,gamma_0]
    else:
        val = np.array((count_reward*freq_print,error_within[:count_within].mean())).reshape((-1,2))
        return [v_0,np.concatenate((mean_reward,val)),v_0_past,nb_past,gamma_0] 