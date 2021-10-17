# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 23:24:54 2019

@author: othmane.mounjid
"""


### Load paths 
import os 
import sys
Path_parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,Path_parent_directory + "\\Plotting")

####### Import liraries 
import numpy as np 
import pandas as pd
  
###############################################################################
###############################################################################
########################################## Start RL functions #################
###############################################################################
###############################################################################

#### Reward after execution
def reward_exec(qsame, bb_pos, gain = 2, cost_out = -1, cost_stay = -0.5):
    if bb_pos ==  0: ## win if execution
        return gain
    elif bb_pos ==  -1: ## cost of a market order
        return cost_out
    else : ## cost of waiting
        return cost_stay

#### Function that encodes the new order book state after a market order
def State_after_market(Lob_state_0,reward_0,reward_exec=reward_exec):
    Lob_state_res = list(Lob_state_0)
    reward_res = 0 # float(reward_0)
    h_0_market = 0
    if (Lob_state_res[1] >= 1): 
        ### Regeneration 
        Lob_state_res[1] = -1
        Lob_state_res[0] =  max(Lob_state_res[0] - 1,0)
        reward_res = reward_exec(Lob_state_res[0],Lob_state_res[1])
    return [Lob_state_res,reward_res,h_0_market]

#### Inner loops  
###### Benchmark inner loop
def Lob_simu_inner_loop1(h_0,h_0_past,h_0_Stay,h_0_Mkt,Lob_state_0,Intens_val,nb_iter, write_option = True, size_q = 2, gamma = 0.1, Error = None, reward_exec = None, eta = 1):
    cols_interest = ['Limit', 'Cancel', 'Market']
    Lob_state_before = list(Lob_state_0) #
    Lob_state = list(Lob_state_0) #
    if write_option:
        df_res = pd.DataFrame(np.zeros((nb_iter+1,3)), columns = ['BB size','BB pos','Event'])
        df_res.loc[0,:] = Lob_state + [None]
    intensities_values = np.zeros(3)
    h_0_cum = np.zeros((size_q,size_q+1))
    h_0_cum_stay = np.zeros((size_q,size_q+1)) ## compute in the same time the value of staying in the lob
    h_0_cum_mkt = np.zeros((size_q,size_q+1)) ## compute in the same time the value of sending a market order
    
    ### Main loop
    for n in range(nb_iter): # n = 0
        #### Get market decision
        Lob_state_before = list(Lob_state)
        index_row_b = Lob_state[0]
        intensities_values[:3] = Intens_val.loc[index_row_b, cols_interest]
        if (Lob_state[0] == 1) and (Lob_state[1] == 1): ## constraint cancellation impossible when i am the LAST one remaining in the queue
            intensities_values[1] = 0
        times = np.random.exponential(1/intensities_values) ## time decision
        index_min = times.argmin() ## index decision
        
        #### Apply market decision
        if index_min == 0: ## Buy Limit order
            Lob_state[0] = min(Lob_state[0] + 1,size_q-1)
            reward = reward_exec(Lob_state[0], Lob_state[1])
        elif index_min == 1: ## Buy Cancel order
            if (Lob_state[0] <= 1) and (Lob_state[1] <= 0) : # index_min = 1
                ### Regeneration 
                Lob_state[0] = max(Lob_state[0] - 1,0)
                reward = 0
            elif (Lob_state[0] > 1) and (Lob_state[1] <= Lob_state[0] - 1)  :
                Lob_state[0] -= 1
                reward = reward_exec(Lob_state[0], Lob_state[1])
            elif (Lob_state[0] > 1) and (Lob_state[1] == Lob_state[0]) :
                Lob_state[0] -= 1
                Lob_state[1] -= 1
                reward = reward_exec(Lob_state[0], Lob_state[1])
            else:
                raise Exception(" Error cancellation impossible")
        elif index_min == 2: ## Sell Market order
            if (Lob_state[1] == 1):
                Lob_state[0] = max(Lob_state[0] - 1,0)
                Lob_state[1] = 0
                reward = reward_exec(Lob_state[0], Lob_state[1])
            elif (Lob_state[1] > 1):
                Lob_state[0] -= 1   
                Lob_state[1] -= 1 
                reward = reward_exec(Lob_state[0], Lob_state[1])
            elif (Lob_state[1] <= 0):
                Lob_state[0] = max(Lob_state[0] - 1,0)
                reward = 0
            else:
                raise Exception(" Error Sell Market order impossible")
                    
        ### the value of stay decision : h_0 stay 
        if (Lob_state[1] <= 0):
            h_0_stay = 0
        else:
            h_0_stay = h_0[Lob_state[0],Lob_state[1]+1] # TRANSLATE BECAUSE OF -1 is an execution by a market order
        
        ### the value of market decision : h_0 market
        Lob_state_mkt,reward_mkt,h_0_market = State_after_market(Lob_state_before,reward,reward_exec=reward_exec)
        
        ### Update the values  h_0, h_0_cum, h_0_past and nb_past
        h_0_cum_stay[Lob_state_before[0],Lob_state_before[1]+1] = (reward + eta*h_0_stay - h_0_Stay[Lob_state_before[0],Lob_state_before[1]+1])
        h_0_cum_mkt[Lob_state_before[0],Lob_state_before[1]+1] = (reward_mkt + eta*h_0_market - h_0_Mkt[Lob_state_before[0],Lob_state_before[1]+1])
        h_0_cum[Lob_state_before[0],Lob_state_before[1]+1] = (max(reward_mkt + eta*h_0_market, reward + eta*h_0_stay) - h_0[Lob_state_before[0],Lob_state_before[1]+1])
       
        ### Update h_0
        h_0[Lob_state_before[0],Lob_state_before[1]+1] += gamma*h_0_cum[Lob_state_before[0],Lob_state_before[1]+1]
        h_0_Stay[Lob_state_before[0],Lob_state_before[1]+1] += gamma*h_0_cum_stay[Lob_state_before[0],Lob_state_before[1]+1]
        h_0_Mkt[Lob_state_before[0],Lob_state_before[1]+1] += gamma*h_0_cum_mkt[Lob_state_before[0],Lob_state_before[1]+1]


        ## update past values 
        h_0_past[Lob_state_before[0],Lob_state_before[1]+1,1] = h_0_cum_stay[Lob_state_before[0],Lob_state_before[1]+1]
        h_0_past[Lob_state_before[0],Lob_state_before[1]+1,0] = h_0_cum_mkt[Lob_state_before[0],Lob_state_before[1]+1] 


        ### Write result
        if write_option:
            df_res.loc[n+1,:] = Lob_state + [index_min]
        
        if Lob_state_before[1]<= 0:
            raise Exception(" Error Lob_state_before[1]<= 0")
        ## Out of the domain
        if (Lob_state[1] <= 0):
            qsame = np.random.randint(1,size_q)
            bb_pos = np.random.randint(1,qsame+1)
            Lob_state_before = [qsame,bb_pos]
            Lob_state = list(Lob_state_before)
    
    error_val = Error(h_0)        
    if write_option:
        return {'lob':Lob_state, 'History':df_res ,'Result':[h_0,h_0_past,error_val,h_0_Stay,h_0_Mkt]}
    else:
        return {'lob':Lob_state, 'History':pd.DataFrame() ,'Result':[h_0,h_0_past,error_val,h_0_Stay,h_0_Mkt]}


#### Super loops
###### Benchmark super loop
def Lob_simu_super_Loop1(nb_episode,Intens_val,nb_iter, gamma= 0.2,Lob_state_0=None,freq_print=100,size_q = 2,Error=None, write_option = False, h_0 = None, reward_exec = None, eta = 1, h_0_Stay = None, h_0_Mkt = None, print_option =  True, pctg_0 = 0.1):
    size_mean = int(nb_episode//freq_print)
    if (h_0 is None) :
        h_0 = np.ones((size_q,size_q+1)) # For bb_pos : index 0 is market order, index 1 is execution 
    if (h_0_Stay is None) :
        h_0_Stay = np.ones((size_q,size_q+1)) # For bb_pos : index 0 is market order, index 1 is execution 
    if (h_0_Mkt is None) :
        h_0_Mkt = np.ones((size_q,size_q+1)) # For bb_pos : index 0 is market order, index 1 is execution 
    h_0_past = np.zeros((size_q,size_q+1,2))
    
    ## FINAL CONSTRAINT
    for qsame in range(size_q): # qsame is the size
        h_0[qsame,qsame+2:] = np.nan
        h_0[qsame,0] = reward_exec(qsame, -1)# market
        h_0[qsame,1] = reward_exec(qsame, 0)# execution
        h_0_Stay[qsame,qsame+2:] = np.nan
        h_0_Mkt[qsame,qsame+2:] = np.nan
        
        
    ## Errors
    error_within = np.zeros(freq_print) ## 
    error_within_estim = np.zeros(freq_print)
    count_within = 0
    count_reward = 0
#    mean_reward = np.zeros((size_mean,2))
#    mean_error_estim = np.zeros((size_mean,2))
#    
#    ## initialisation pass
#    gamma_0 = float(gamma)

    mean_reward = np.zeros((size_mean+1,2))
    mean_error_estim = np.zeros((size_mean+1,2))
    
    ## initialisation pass
    gamma_0 = float(gamma)
    ## initialisation of the error
    mean_reward[count_reward] = Error(h_0)
    count_reward += 1
    
    ### Parameters ; upper level 
    h_0_before = np.array(h_0)
    h_0_Stay_before = np.array(h_0_Stay)
    h_0_Mkt_before = np.array(h_0_Mkt) 
    h_0_past_before = np.array(h_0_past)
    count_period =0   
    
    for ep in range(nb_episode):
        h_0,h_0_past,error_val,h_0_Stay,h_0_Mkt = Lob_simu_inner_loop1(h_0,h_0_past,h_0_Stay,h_0_Mkt,Lob_state_0,Intens_val,nb_iter, write_option = write_option, size_q = size_q, gamma = gamma_0, Error = Error, reward_exec = reward_exec, eta = eta)['Result']
        error_within[count_within] = error_val
        error_within_estim[count_within] = np.linalg.norm(h_0_past)
        count_within += 1
        if (((ep %(freq_print))==freq_print-1) and (ep> 0)):
            if print_option:
                print(" frequency is : " + str(ep))
            mean_reward[count_reward] = (count_reward*freq_print,error_within.mean())
            mean_error_estim[count_reward] = (count_reward*freq_print,error_within_estim.mean())
            ## update gamma :
            index_count_before = max(count_reward-1,1)#0
            pctg_last = ((mean_error_estim[index_count_before,1] - mean_error_estim[count_reward,1])/mean_error_estim[index_count_before,1])
            if (pctg_last <= pctg_0) and (count_reward >= 1) and (count_period >= 3):
                h_0 = np.array(h_0_before)
                h_0_Stay = np.array(h_0_Stay_before)
                h_0_Mkt = np.array(h_0_Mkt_before) 
                h_0_past = np.array(h_0_past_before)                
                
                gamma_0 = max(gamma_0/2,0.01) # count_gamma_0 = 1
                count_period = 0
            else : 
                h_0_before = np.array(h_0)
                h_0_Stay_before = np.array(h_0_Stay)
                h_0_Mkt_before = np.array(h_0_Mkt)
                h_0_past_before = np.array(h_0_past)                
            error_within[:] = 0
            count_within = 0
            count_reward += 1
            count_period += 1
    if (count_within==0):
        return  [h_0,h_0_past,mean_reward,h_0_Stay,h_0_Mkt]
    else:
        val = np.array((count_reward*freq_print,error_within[:count_within].mean())).reshape((-1,2))
        return [h_0,h_0_past,np.concatenate((mean_reward,val)),h_0_Stay,h_0_Mkt]      