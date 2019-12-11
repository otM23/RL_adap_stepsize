# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 10:18:35 2019

@author: othmane.mounjid
"""

### Import path 

### Import libraries 
def prox_id(x):
  return x
  
def prox_1(x,_eta):
  if x>=0:
      return x - _eta
  else:
      return x + _eta

def prox_2(x,_eta):
  return x/(1 + 2*_eta)