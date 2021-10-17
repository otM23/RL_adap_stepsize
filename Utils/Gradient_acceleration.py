# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 10:25:21 2019

@author: othmane.mounjid
"""

### Import libraries
import numpy as np

   
def SAGA_memory(index_state,h,diff_h,diff_h_past,weights,nb_past,n_max,prox,discount = 0.2):
      j = nb_past[index_state]
      nb_values = weights[:].sum()
      h[index_state] = h[index_state] + discount*(diff_h[index_state] - diff_h_past[index_state][j]  + ((diff_h_past[index_state][:]*weights).sum())/(nb_values))
      h[index_state] = prox(h[index_state])
      if (n_max > j +1):
          diff_h_past[index_state][j] = diff_h[index_state] 
      else:
          diff_h_past[index_state][:j] = diff_h_past[index_state][1:n_max]
          diff_h_past[index_state][j] = diff_h[index_state]
      nb_past[index_state] = min(nb_past[index_state]+1,n_max-1)

def SAGA_memory_last(index_state,h,diff_h,diff_h_past,weights,nb_past,n_max,prox,discount = 0.2):
      j = nb_past[index_state]
      nb_values = max(weights[:j].sum(),1)
      h[index_state] = h[index_state] + discount*(diff_h[index_state] - diff_h_past[index_state][j]  + ((diff_h_past[index_state][:j]*weights[:j]).sum())/(nb_values))
      h[index_state] = prox(h[index_state])
      if (n_max > j +1):
          diff_h_past[index_state][j] = diff_h[index_state] 
      else:
          diff_h_past[index_state][:j] = diff_h_past[index_state][1:n_max]
          diff_h_past[index_state][j] = diff_h[index_state]
      nb_past[index_state] = min(nb_past[index_state]+1,n_max-1)

def SAG_rnd(index_state,h,diff_h,diff_h_past,weights,nb_past,n_max,prox,discount = 0.2):
      j = np.random.randint(n_max)
      nb_past[index_state,j] = 1 # min(self._nb_past[i]+1,self._n_max)
      nb_values = max(nb_past[index_state,:].sum(),1)
      diff_h_past[index_state,j] = diff_h[index_state] 
      h[index_state] = h[index_state] + discount*(((diff_h_past[index_state,:]).sum())/(nb_values))
      h[index_state] = prox(h[index_state])
      nb_past[index_state,j] = 1
      
def SAGA_rnd(index_state,h,diff_h,diff_h_past,weights,nb_past,n_max,prox,discount = 0.2):
      j = np.random.randint(n_max)
      nb_values = max(nb_past[index_state,:].sum(),1)
      h[index_state] = h[index_state] + discount*(diff_h[index_state] - diff_h_past[index_state,j]  + ((diff_h_past[index_state,:]).sum())/(nb_values))
      h[index_state] = prox(h[index_state])
      diff_h_past[index_state,j] = diff_h[index_state] 
      nb_past[index_state,j] = 1
      
### Loop version
def SAGA_memory_loop(index_state,h,diff_h,diff_h_past,weights,nb_past,n_max,prox,discount = 0.2,nb_elt=1):
      j = nb_past[index_state]
      nb_values = weights[:].sum()
      for elt in range(nb_elt):
          h[elt][index_state] = h[elt][index_state] + discount*(diff_h[elt][index_state] - diff_h_past[elt][index_state,j]  + ((diff_h_past[elt][index_state,:]*weights).sum())/(nb_values))
          h[elt][index_state] = prox[elt](h[elt][index_state])
          if (n_max > j +1):
              diff_h_past[elt][index_state,j] = diff_h[elt][index_state] 
          else:
              diff_h_past[elt][index_state,:j] = diff_h_past[elt][index_state,1:n_max]
              diff_h_past[elt][index_state,j] = diff_h[elt][index_state]
      nb_past[index_state] = min(nb_past[index_state]+1,n_max-1)

def SAGA_memory_last_loop(index_state,h,diff_h,diff_h_past,weights,nb_past,n_max,prox,discount = 0.2,nb_elt=1):
      j = nb_past[index_state]
      nb_values = max(weights[:j].sum(),1)
      for elt in range(nb_elt):
          h[elt][index_state] = h[elt][index_state] + discount*(diff_h[elt][index_state] - diff_h_past[elt][index_state,j]  + ((diff_h_past[elt][index_state,:j]*weights[index_state,:j]).sum())/(nb_values))
          h[elt][index_state] = prox[elt](h[elt][index_state])
          if (n_max > j +1):
              diff_h_past[elt][index_state,j] = diff_h[elt][index_state] 
          else:
              diff_h_past[elt][index_state,:j] = diff_h_past[elt][index_state,1:n_max]
              diff_h_past[elt][index_state,j] = diff_h[elt][index_state]
      nb_past[index_state] = min(nb_past[index_state]+1,n_max-1)

def SAG_rnd_loop(index_state,h,diff_h,diff_h_past,weights,nb_past,n_max,prox,discount = 0.2,nb_elt=1):
      j = np.random.randint(n_max)
      nb_past[index_state,j] = 1 # min(self._nb_past[i]+1,self._n_max)
      nb_values = max(nb_past[index_state,:].sum(),1)
      for elt in range(nb_elt):
          diff_h_past[elt][index_state,j] = diff_h[elt][index_state] 
          h[elt][index_state] = h[elt][index_state] + discount*(((diff_h_past[elt][index_state,:]).sum())/(nb_values))
          h[elt][index_state] = prox[elt](h[elt][index_state])
      nb_past[index_state,j] = 1
      
def SAGA_rnd_loop(index_state,h,diff_h,diff_h_past,weights,nb_past,n_max,prox,discount = 0.2,nb_elt=1):
      j = np.random.randint(n_max)
      nb_values = max(nb_past[index_state,:].sum(),1)
      for elt in range(nb_elt):
          h[elt][index_state] = h[elt][index_state] + discount*(diff_h[elt][index_state] - diff_h_past[elt][index_state,j]  + ((diff_h_past[elt][index_state,:]).sum())/(nb_values))
          h[elt][index_state] = prox[elt](h[elt][index_state])
          diff_h_past[elt][index_state,j] = diff_h[elt][index_state] 
      nb_past[index_state,j] = 1
