#!/usr/bin/env python
# coding: utf-8

# The script is for the test of parameter settings. Take the initial parameters and initial samples and 
# run the MCMC samples on the |psi|^2 distrbution and obtain the energy estimate. 



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
    
import jax.numpy as jnp
from jax import grad
from jax import hessian
import numpy as np
import jax.random as random
from jax import pmap
from jax import vmap
from jax import jit
import jax.ops
import time
from jax import config
config.update("jax_enable_x64", True)
from jax.scipy.special import logsumexp
import pathlib
from tempfile import TemporaryFile
import matplotlib.pyplot as plt


# $\newcommand{\bt}{{\boldsymbol{\theta}}}$
# $\newcommand{\norm}[1]{\left\Vert {#1}\right\Vert}$
# 
# $$\bt'=\arg\min_{\bt',V} \frac{1}{2}\sum_{a=1}^M\norm{\frac{\psi^a_{\bt'}}{\norm{\psi^a_{\bt'}}_\mu }-\frac{(I-\epsilon H)\psi^a_\bt}{ \norm{(I-\epsilon H)\psi^a_\bt}_\mu } }_\mu^2$$

# In[2]:


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
logansatz2 = vmap(jlogansatz, (None, 0, 0), 0)
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

@jit
def queryHx_x_more(state, fftW0, b0,logxstate,dlogxstate_dweights):
    sames= state ^ state[(jnp.arange(d) + 1) % d]
    energy1 = 2*jnp.sum(sames)-d
    Hx1_x = -delta*energy1
    Hdx1_x= -delta*energy1*dlogxstate_dweights
     
    statesxy = jnp.repeat(jnp.expand_dims(state, -1), d, -1).T
    i0 = jnp.arange(d)
    
    statesxy = statesxy.at[i0,i0].set(~statesxy[i0,  i0])
    statesxy = statesxy.at[i0, (i0+1)%d].set(~statesxy[i0, (i0+1)%d])
    
    #sames=~sames
    
    flips = (2*jnp.sum(statesxy, axis = -1) + statesxy[..., 0] > d)
    statesxy = statesxy ^ jnp.expand_dims(flips, -1)
    
    logxstatesxy,dlogxstatesxy=jcompute(statesxy, fftW0, b0)
    wave_ratios=jnp.exp(logxstatesxy-logxstate)
    sameswave_ratios=sames*wave_ratios
    Hx2_x = -2*jnp.sum(sameswave_ratios)    
    Hdx2_x= -2*jnp.sum(dlogxstatesxy*sameswave_ratios.reshape(-1,1),axis=0) 
    return Hx1_x+Hx2_x,Hdx1_x+Hdx2_x

queryHx_x_more1 = vmap(queryHx_x_more, (0, None, None,0,0), 0)
jqueryHx_x_more = jit(queryHx_x_more1)

@jit
def queryHx_x(state, fftW0, b0,logxstate):
    sames= state ^ state[(jnp.arange(d) + 1) % d]
    energy1 = 2*jnp.sum(sames)-d
    Hx1_x = -delta*energy1
     
    statesxy = jnp.repeat(jnp.expand_dims(state, -1), d, -1).T
    i0 = jnp.arange(d)
    
    statesxy = statesxy.at[i0,i0].set(~statesxy[i0,  i0])
    statesxy = statesxy.at[i0, (i0+1)%d].set(~statesxy[i0, (i0+1)%d])
    
    #sames=~sames
    
    flips = (2*jnp.sum(statesxy, axis = -1) + statesxy[..., 0] > d)
    statesxy = statesxy ^ jnp.expand_dims(flips, -1)
    
    logxstatesxy,dlogxstatesxy=jcompute(statesxy, fftW0, b0)
    wave_ratios=jnp.exp(logxstatesxy-logxstate)
    sameswave_ratios=sames*wave_ratios
    Hx2_x = -2*jnp.sum(sameswave_ratios)    
    return Hx1_x+Hx2_x

queryHx_x1 = vmap(queryHx_x, (0, None, None,0), 0)
jqueryHx_x = jit(queryHx_x1)
queryHx_x2 = vmap(jqueryHx_x, (None, 0, 0,0), 0)
jjqueryHx_x = jit(queryHx_x2)
 
@jit
def cvcompute(state):
    sames= state ^ state[(jnp.arange(d) + 1) % d]
    energy1 = 2*jnp.sum(sames)-d
    return -energy1/d
cvcompute1 = vmap(cvcompute, 0, 0)
jcvcompute= jit(cvcompute1) 

@jit
#with Eloc_i, used for RGN
def sample_more(inputs,i):
    inputs,store_flipornot=jax.lax.scan(update,inputs,None,d)
    paccflip=jnp.mean(store_flipornot, axis=0)
    key,states,cvstates,fftW0,b0=inputs
    logxstates, dlogx_dweights=jcompute(states,fftW0,b0)
    Hxstates,Hdxstates=jqueryHx_x_more(states,fftW0,b0,logxstates, dlogx_dweights)
    cvstates=jcvcompute(states)
    return (key,states,cvstates,fftW0,b0),(states,logxstates, dlogx_dweights, Hxstates,Hdxstates,cvstates)
@jit
def update(inputs,i):
    key,states,cvstates,fftW0,b0=inputs
    key, key1,key2 = random.split(key, num=3)
#     xstates=jansatzcompute_less(states,W0,b0)
    i0=jnp.arange(parallel)
    perturbs=states
    
    select = random.uniform(key1, shape = (parallel,))*(d // 2)
    ups = jnp.cumsum(perturbs, axis = -1)
    i1 = jnp.argmax(jnp.greater(ups, select[:, jnp.newaxis]), axis = -1)
    select = random.uniform(key2, shape = (parallel,))*(d // 2)
    downs = jnp.cumsum(~perturbs, axis = -1)
    i2 = jnp.argmax(jnp.greater(downs, select[:, jnp.newaxis]), axis = -1)
    # perturbs = jax.ops.index_update(perturbs, jax.ops.index[i0, i1], ~perturbs[i0, i1])
    # perturbs = jax.ops.index_update(perturbs, jax.ops.index[i0, i2], ~perturbs[i0, i2])
    
    perturbs=  perturbs.at  [i0, i1] .set( ~perturbs[i0, i1])
    perturbs = perturbs.at  [ i0, i2] .set( ~perturbs[i0, i2])
    
    flips = (2*jnp.sum(perturbs, axis = -1) + perturbs[..., 0] > d)
    perturbs = perturbs ^ jnp.expand_dims(flips, -1)

    logxperturbs=jlogansatz(perturbs,fftW0,b0)
    logxstates=jlogansatz(states,fftW0,b0)

    flipornot=random.exponential(key,(parallel,))
    flipornot=(-0.5*flipornot<logxperturbs.real-logxstates.real)
    #flipornot=(-0.5*flipornot<inv_temp*(logxperturbs.real-logxstates.real))

    states=jax.numpy.where(jnp.reshape(flipornot, (-1,1)),perturbs,states)
    return (key,states,cvstates,fftW0,b0),flipornot

def Computevecmean(dist,  fmcmc, fmixin, beta):
    return beta*(dist*fmcmc).sum(axis=0)+(1-beta)*fmixin.mean(axis=0)

def printparameter():
    print("d=",d)
    print("alpha=", alpha)
    print("delta=", delta)
    print("parallel=",parallel)
    print("iterations=",iterations)
    print("T=",T)
    print("myseed=",myseed)
    print("test_iter=",test_iter)
    print("inv_temp=",inv_temp)
setting=sys.argv
d=int(setting[1])
parallel=int(setting[2])
myseed=int(setting[3])
iterations=int(setting[4])
delta=float(setting[5])
alpha=int(setting[6])
test_iter=int(setting[7])
inv_temp=float(setting[8])
batchsize=2000

save_dir=f"/scratch/hz1994/vinla/0610/d_{d}/alpha50_{inv_temp:.2f}_{myseed}_split0.50_test/"
pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
weights_filename=f"/scratch/hz1994/vinla/vs_temp_mix/d_200/alpha50_{inv_temp:.2f}_{myseed}_split0.50/w_list{myseed}_L1_eps0.100_eps1_0.100_delta_1.0.npy"
states_filename=f"/scratch/hz1994/vinla/vs_temp_mix/d_200/alpha50_{inv_temp:.2f}_{myseed}_split0.50/states_list{myseed}_L1_eps0.100_eps1_0.100_delta_1.0_iter5000.npy"

T=20
sr_reg=0.001
# mixedin=parallel*T
 
# E0=xxz_exact(d, delta)
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

E_list=[]
std_list=[]
w_list=[]
Hloc_list=[]
E_inner=[]
g_list=[]
states_list=[]
print("weights_filename:", weights_filename, "test_iter:",test_iter)    
if np.load(states_filename).shape[0]==100:
    test_iter=-1

weights=np.load(weights_filename)[test_iter]

key = random.PRNGKey(myseed)
W0=jnp.reshape(weights[:-alpha],(alpha,d))
fftW0=jnp.fft.fft(W0)
b0=jnp.reshape(weights[-alpha:],(alpha,1))

states=(np.load(states_filename)[test_iter].reshape(T,parallel,d))[-1]
cvstates=jcvcompute(states)

printparameter()
    
def batch_queryHx_x(samples,nsamples,fftW0,b0,logsamples,batchsize):
    Hloc=np.zeros((nsamples,1)).astype(complex)
    for i in range(nsamples//batchsize+1):
        lower=i*batchsize
        upper=np.minimum((i+1)*batchsize, nsamples)
        ind=np.arange(lower,upper)
        batchHloc=jqueryHx_x(samples[ind], fftW0, b0,logsamples[ind])[:,None]
        Hloc[ind]=batchHloc
    return Hloc
    
def computeS_batch (dlogsamples,nsamples,batchsize):
    S=np.zeros((dlogsamples.shape[1], dlogsamples.shape[1])).astype(complex)
    for i in range(nsamples//batchsize+1):
        lower=i*batchsize
        upper=np.minimum((i+1)*batchsize, nsamples)
        ind=np.arange(lower,upper)
        batch_dlog=dlogsamples[ind]
        S=S+ jnp.matmul(batch_dlog.conj().T,batch_dlog)
    S=S/dlogsamples.shape[0]
    return S

def computeg_batch (dlogsamples,Hloc, nsamples,batchsize):
    g=np.zeros((dlogsamples.shape[1],)).astype(complex)
    for i in range(nsamples//batchsize+1):
        lower=i*batchsize
        upper=np.minimum((i+1)*batchsize, nsamples)
        ind=np.arange(lower,upper)
        batch_dlog=dlogsamples[ind]
        batch_Hloc=Hloc[ind]
        gbatch= jnp.sum(batch_dlog.conj()*batch_Hloc,axis=0)
        g=g+ gbatch
    g=g/dlogsamples.shape[0]
    return g


for iteration in range(iterations):
    inputs=(key,states,cvstates,fftW0,b0)

    (key,states,cvstates,__,__),(store_states,store_logxstates,store_dlogx_dweights,Hloc_raw,\
            Eloci,store_cvstates)=jax.lax.scan(sample_more,inputs,None,T)

    Hloc=jnp.reshape(Hloc_raw,(-1,1))
    energy=jnp.mean(Hloc_raw).real/d-E0
    E_list.append(energy)
    print("%d  energy=%12.6e"%(iteration, energy))
    states_list.append(store_states)

    if iteration % 100 ==0:
        np.save(save_dir+'E_list%d_%.2f_iter_%d.npy'%(myseed,inv_temp,test_iter) ,E_list)
        np.save(save_dir+'states%d_%.2f_iter_%d.npy'%(myseed,inv_temp,test_iter) ,states_list)

    if iteration % 10==0:    
        if np.isnan(E_list[-1]):
            print("iteration=%d, find energy estimate nan"%iteration)
            break
