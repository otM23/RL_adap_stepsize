# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 16:53:02 2019

@author: othmane.mounjid
"""

### Import libraries 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def Plot3D(Resx,Resy,Resz,bins,xlabel='Ask size',ylabel='Bid size',zlabel='Joint distribution',option="save",path ="",ImageName="",xtitle="",
            elev0= 30, azim0=40, dist0= 12,optionXY =1,figsize_ = (8,11), x_tickslabels =  False, x_ticksvalues = np.zeros(1) ):
    if type(bins) == int:
        bins = [bins,bins]
    xpos1 = np.zeros(bins[0]*bins[1])
    ypos1 = np.zeros(bins[0]*bins[1])
    zpos1 = np.zeros(bins[0]*bins[1])
    if optionXY == 1:
        for  i in range(bins[0]):
            for j in range(bins[1]):
                xpos1[i*bins[1]+j] = (Resx[i]+Resx[i+1])/2
                ypos1[i*bins[1]+j] = (Resy[j]+Resy[j+1])/2
                zpos1[i*bins[1]+j] = Resz[i,j]
    if optionXY == 2:
        for  i in range(bins[0]):
            for j in range(bins[1]):
                xpos1[i*bins[1]+j] = Resx[i]
                ypos1[i*bins[1]+j] = Resy[j]
                zpos1[i*bins[1]+j] = Resz[i*bins[1]+j]
        
    fig = plt.figure(figsize = figsize_)
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev= elev0,azim=azim0)
    ax.dist=dist0  
    if x_tickslabels:
        ticks_values = tuple([np.format_float_scientific(elt, unique=False, precision=2) for elt in x_ticksvalues])
        ax.set_xticklabels(list(ticks_values))
    ax.plot_trisurf(xpos1, ypos1, zpos1, linewidth=0.2, antialiased=True,
                    cmap=plt.cm.rainbow)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    #ax.set_zlim((0,0.0035))
    plt.grid()
    if option == "save" :
        plt.savefig(path+ImageName+".pdf", bbox_inches='tight')  
    plt.show()

#### PLot 2d values :
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

def Plot_sns(df,option=False,path ="",ImageName="",xtitle="", xlabel ="", ylabel ="", annot =True , fig = False, a = 0, b = 0, subplot0 = 0, cbar = True, cmap="PuBu", mask = None, fmt = '.2g', Nset_tick_x = False, Nset_tick_y = False, annot_size = 20):
    if not fig:
        ax = plt.axes()
    else:
        ax = fig.add_subplot(a,b,subplot0)
    if Nset_tick_x :
        ax.get_xaxis().set_visible(False)
    if Nset_tick_y :
        ax.get_yaxis().set_visible(False)
    
    ax = sns.heatmap(df,cmap=cmap, ax = ax, annot = annot,cbar = cbar,mask=mask, fmt= fmt, annot_kws={"size": annot_size})
    ax.set_title(xtitle,fontsize = 18)
    ax.set_xlabel(xlabel,fontsize = 18)
    ax.set_ylabel(ylabel,fontsize = 18)
    ax.invert_yaxis()
    if option == "save" :
        plt.savefig(path+ImageName+".pdf", bbox_inches='tight') 

def Plot_sns_2(df,option=False,path ="",ImageName="",xtitle="", xlabel ="", ylabel ="", annot =True, fig = False, a = 0, b = 0, subplot0 = 0, cbar = True, cmaps = [], masks = [], fmt = '.2g', Nset_tick_x = False, Nset_tick_y = False, annot_size = 20):
    if not fig:
        ax = plt.axes()
    else:
        ax = fig.add_subplot(a,b,subplot0)
        
    if Nset_tick_x :
        ax.get_xaxis().set_visible(False)
    if Nset_tick_y :
        ax.get_yaxis().set_visible(False)
    
    counter = 0 
    for mask0 in masks: # mask0 = masks[0]
        sns.heatmap(df,cmap=cmaps[counter],ax=ax,annot = annot, cbar = cbar, mask = mask0, fmt = fmt, annot_kws={"size": annot_size})
        counter += 1 
    ax.set_title(xtitle,fontsize = 18)
    ax.set_xlabel(xlabel,fontsize = 18)
    ax.set_ylabel(ylabel,fontsize = 18)
    ax.invert_yaxis()
    if option == "save" :
        plt.savefig(path+ImageName+".pdf", bbox_inches='tight') 
        