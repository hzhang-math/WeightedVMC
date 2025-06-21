#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 22:16:28 2023

@author: hzhang
"""

import os
import math
import sys
import copy
import scipy
import pickle

from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import spsolve
from numpy.linalg import norm
from numpy.linalg import qr
from  scipy.linalg import  svdvals
import jax.numpy as jnp
import numpy as np
import jax.random as random
from jax import pmap
from jax import vmap
from jax import jit
import jax.ops
import time
# from jax.config import config
# config.update("jax_enable_x64", True)
from jax.scipy.special import logsumexp
import pathlib
from tempfile import TemporaryFile
import matplotlib.pyplot as plt
from functools import partial
import pylab as pl
from IPython import display
#%%
@jit 
def logansatz(s,fftW0,b0):
    Fsigma=jnp.fft.fft(2*s-1)
    theta=jnp.fft.ifft(fftW0*jnp.conj(Fsigma))+b0
    logx_sigma=jnp.sum(jnp.log(jnp.cosh(theta)))
#     x_sigma=jnp.exp(logx_sigma)
    return logx_sigma
logansatz1 = vmap(logansatz, (0, None, None), 0)
jlogansatz= jit(logansatz1)
logansatz2 = vmap(jlogansatz, (0, 0, 0), 0)
jjlogansatz = jit(logansatz2)

@jit
def compute(s,fftW0,b0):
    Fsigma=jnp.fft.fft(2*s-1)
    theta=jnp.fft.ifft(fftW0*jnp.conj(Fsigma))+b0
    logx_sigma=jnp.sum(jnp.log(jnp.cosh(theta)))
    tanhtheta=jnp.tanh(theta)
    dlogx_sigma_db=jnp.sum(tanhtheta,axis=-1)
    dlogx_sigma_dw=jnp.fft.ifft(jnp.fft.fft(tanhtheta)*Fsigma)
    dlogx_dweights=jnp.concatenate((dlogx_sigma_dw.flatten(),jnp.reshape(dlogx_sigma_db,(-1,))))
    return logx_sigma, dlogx_dweights
compute1 = vmap(compute, (0, None, None), 0)
jcompute = jit(compute1)
compute2 = vmap(jcompute, (None, 0, 0), 0)
jjcompute = jit(compute2)
compute3 = vmap(jjcompute, (0, None, None), 0)
jjjcompute = jit(compute3)

compute_respective = vmap(jcompute, (0, 0,0), 0)
jcompute_respective = jit(compute_respective)
@jit
def queryHx_x(state, fftW0, b0,logxstate):
    sames= state ^ state[(jnp.arange(d) + 1) % d]
    energy1 = 2*jnp.sum(sames)-d
    Hx1_x = -delta*energy1
     
    statesxy = jnp.repeat(jnp.expand_dims(state, -1), d, -1).T
    i0 = jnp.arange(d)
    
    statesxy = statesxy.at[i0,i0].set(~statesxy[i0,  i0])
    statesxy = statesxy.at[i0, (i0+1)%d].set(~statesxy[i0, (i0+1)%d])

    logxstatesxy,dlogxstatesxy=jcompute(statesxy, fftW0, b0)
    wave_ratios=jnp.exp(logxstatesxy-logxstate)
    sameswave_ratios=sames*wave_ratios
    Hx2_x = -2*jnp.sum(sameswave_ratios)    
    return Hx1_x+Hx2_x

queryHx_x1 = vmap(queryHx_x, (0, None, None,0), 0)
jqueryHx_x = jit(queryHx_x1)

@partial(jit, static_argnums=1)
def cvcompute(state,d):
    sames= state ^ state[(jnp.arange(d) + 1) % d]
    energy1 = 2*jnp.sum(sames)-d
    return -energy1/d
cvcompute1 = vmap(cvcompute, (0,None), 0)
jcvcompute= jit(cvcompute1) 

@jit
def querySx_x(state, fftW0, b0,logxstate):
    sames= state ^ state[(jnp.arange(d) + 1) % d]
    energy1 = 2*jnp.sum(sames)-d
    Hx1_x = -delta*energy1
    return Hx1_x
querySx_x1 = vmap(querySx_x, (0, None, None,0), 0)
jquerySx_x = jit(querySx_x1)

def get_next_color(ax):
        # Access the prop_cycler and get the next color from the color cycle
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        color_index = len(ax.lines) % len(colors)
        return colors[color_index]


delta=1

def plot(ax, Hloc_error_list,cv_unique,color=None,label=None): 
    if not color:
        color=get_next_color(ax)   
    bp=ax.boxplot(Hloc_error_list,positions=cv_unique,showbox =True,\
                         showfliers=False,widths=0.02,whis=1.5,\
                         boxprops= dict(linewidth=1.5, color=color),
                         medianprops=dict(color='black', linewidth=1.5),
             whiskerprops=dict(linestyle='-',linewidth=1.5, color=color))
    for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
            plt.setp(bp[item], color=color)
    
    ax.set_yscale("log")

    cvplot=np.array([-1,-0.5,0,0.5,1])
    cvticklabels=[str(cv) for cv in cvplot]
    ax.set_xticks(cvplot)
    ax.set_xticklabels(cvticklabels)
    ax.set_ylabel(r"$|H\psi/\psi-E_0|$")    
    ax.set_xlabel(r"$s$")   
    ax.set_xlim([-1,1])
    ax.set_ylim([1e-6,1e1])
    return ax

def Hloc_boxplot(ax, fftW0, b0, color=None,label=None,d=100,alpha=5):
    if d==100:
        if delta==0.5:
            E0=-1.5002721151896314
        elif delta==1.0:
            E0=-1.772918283003674
        elif delta==1.5:
            E0=-2.093819885253906
        else:
            print("no defined E0")
    if d==200:
        if delta==0.5:
            E0=-1.5000680204043535
        elif delta==1.0:
            E0=-1.772671065044832
        elif delta==1.5:
            E0=-2.093740234375
        else:
            print("no defined E0")
    pool_states_list=np.load('/scratch/hz1994/vinla/pool/pool_cvunif_d%d.npy'%d)
    pool_states_list=pool_states_list[:2000]
    cvstates=jcvcompute(pool_states_list,d)
    logx_states_list=jlogansatz(pool_states_list,fftW0,b0 )
    Hloc_list=jqueryHx_x(pool_states_list,fftW0,b0,logx_states_list ).real
    
    cv_unique=np.unique(cvstates)
    Hloc_error_list=[] 
    for i, cv in enumerate(cv_unique):
        Hloc_error_list .append(abs(Hloc_list[cvstates==cv]/d-E0)) 
    
    Hlocerrormean=[np.mean(item) for item in Hloc_error_list ]
    ax=plot(ax,Hloc_error_list,cv_unique,color)
    return ax,  cv_unique, Hlocerrormean,Hloc_error_list