#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
    #flipornot=(-0.5*flipornot<logxperturbs.real-logxstates.real)
    flipornot=(-0.5*flipornot<inv_temp_vec*(logxperturbs.real-logxstates.real))

    states=jax.numpy.where(jnp.reshape(flipornot, (-1,1)),perturbs,states)
    return (key,states,cvstates,fftW0,b0),flipornot

def Computevecmean(dist,  fmcmc, fmixin, beta):
    return beta*(dist*fmcmc).sum(axis=0)+(1-beta)*fmixin.mean(axis=0)

def printparameter():
    print("L=",L)
    print("epsilon=",epsilon_max)
    print("epsilon1=",epsilon1)
    print("d=",d)
    # print("beta=",beta)
    print("alpha=", alpha)
    print("delta=", delta)
    print("parallel=",parallel)
    print("iterations=",iterations)
    print("T=",T)
    print("myseed=",myseed)
    # print("save_dir:", save_dir)
    # print("mixedin=", mixedin)
    # print("mixdist=",mixdist)
    print("batchsize=",batchsize)
    print("inv_temp=",inv_temp)
    print("split_temp_fraction=",split_temp_fraction)
setting=sys.argv
L=int(setting[1])
epsilon_max=float(setting[2])
epsilon1=float(setting[3])
d=int(setting[4])
parallel=int(setting[5])
myseed=int(setting[6])
iterations=int(setting[7])
delta=float(setting[8])
alpha=int(setting[9])
inv_temp=float(setting[10])
split_temp_fraction=float(setting[11])

batchsize=1000

# save_dir = f"/scratch/hz1994/vinla/d_{d}/parallel_{parallel}/myseed_{myseed}/alpha_{alpha}/inv_temp_{inv_temp:.2f}/split_temp_fraction_{split_temp_fraction:.2f}/"

# save_dir=str(f"/scratch/hz1994/vinla/parallel{parallel}/"+"alpha%d_%.2f_%d_split%.2f/"%(alpha,inv_temp,myseed,split_temp_fraction))
save_dir=str("./vs_temp_mix/parallel_%d/d_%d/alpha%d_%.2f_%d_split%.2f/"%(parallel,d,alpha,inv_temp,myseed,split_temp_fraction))
pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

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

mu_dist=np.ones( (parallel*T,1) )
mu_dist=mu_dist/np.sum(mu_dist)
logmu_dist=jnp.log(mu_dist)

sr_max=epsilon_max
sr_min=0.01
iterations0=1000
epsilon_list = sr_min*(sr_max/sr_min)**np.arange(2*iterations/iterations0, step = 2/iterations0)
epsilon_list[epsilon_list > sr_max] = sr_max
epsilons_reset = np.copy(epsilon_list)

    
    
key = random.PRNGKey(myseed)
key,key1,key2= random.split(key, num=3)
weights_save = .001*random.normal(key1, shape = (alpha*(d + 1),))     + .001j*random.normal(key2, shape = (alpha*(d + 1),))
weights=jnp.asarray(weights_save)
W0=jnp.reshape(weights[:-alpha],(alpha,d))
fftW0=jnp.fft.fft(W0)
b0=jnp.reshape(weights[-alpha:],(alpha,1))

states_old = jnp.tile(jnp.array([True, False]), d//2)
key3 = random.split(key, num = parallel)
states_save = vmap(random.permutation, in_axes = (0, None), out_axes = 0)(key3, states_old)
flips = (2*jnp.sum(states_save, axis = -1) + states_save[..., 0] > d)
states_save = states_save ^ jnp.expand_dims(flips, -1)
states_save = jnp.reshape(states_save, (parallel, -1))
states = jnp.array(states_save)
cvstates=jcvcompute(states)

split_temp_ind=round(parallel*split_temp_fraction)
print("split_temp_ind=",split_temp_ind)
inv_temp_vec=np.ones(parallel,)
inv_temp_vec[split_temp_ind:]=inv_temp 

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
    epsilon=epsilon_list[iteration]
    w_list.append(copy.deepcopy(weights))
    W0=jnp.reshape(weights[:-alpha],(alpha,d))
    fftW0=jnp.fft.fft(W0)
    b0=jnp.reshape(weights[-alpha:],(alpha,1))

    inputs=(key,states,cvstates,fftW0,b0)

    (key,states,cvstates,__,__),(store_states,store_logxstates,store_dlogx_dweights,Hloc_raw,\
            Eloci,store_cvstates)=jax.lax.scan(sample_more,inputs,None,T)

    cvmcmc=jnp.reshape(store_cvstates,(-1,1))
    logmcmc=jnp.reshape(store_logxstates,(-1,1))
    dlogmcmc=jnp.reshape(store_dlogx_dweights,(-1,alpha*(d+1)))
    Hlocmcmc=jnp.reshape(Hloc_raw,(-1,1))


    key, key1, = random.split(key, num=2)
    store_states=store_states.reshape((-1,d))
    

    vmean=jnp.mean(dlogmcmc,axis=0)
    Ehat=jnp.mean(Hlocmcmc)

    Hlocmcmc=Hlocmcmc-Ehat
    dlogmcmc=dlogmcmc-vmean
    
    S=computeS_batch(dlogmcmc,T*parallel, batchsize)
    g=computeg_batch(dlogmcmc,Hlocmcmc, T*parallel, batchsize)

    regular = (S + sr_reg*jnp.eye(g.size))
    move = -jnp.linalg.solve(regular, g)
    move_norm=jnp.sum(jnp.abs(move)**2)**.5

    energy=jnp.mean(Hloc_raw[:,:split_temp_ind]).real/d-E0
    energy_std=jnp.std(Hloc_raw)
    std_list.append(energy_std)
    E_list.append(energy)
    
    # make a move
    move_norm=jnp.sum(jnp.abs(move)**2)**.5
    guidance=move_norm*epsilon
    weights=weights+move*epsilon1*epsilon

    states_list.append(store_states)

    if iteration % 100 ==0:
        print("%d std=%12.6e energy=%12.6e"%(iteration, energy_std,energy))
        np.save(save_dir+'std_list.npy',std_list)
        np.save(save_dir+'E_list.npy' ,E_list)
        np.save(save_dir+'w_list.npy',w_list)
        np.save(save_dir+'states_list.npy',states_list)
        states_list=[]
        print(f"iteration ={iteration}, save std_list, E_list, w_list and states_list in save_dir {save_dir}")

    if iteration % 10==0:    
        if np.isnan(std_list[-1]):
            print("iteration=%d, find energy estimate nan"%iteration)
            break
    


print("Starts testing")
iterations_test = 2001
T=20
parallel=100
states=states[:100]
print("states.shape",states.shape)
E_list_test = []
Hloc_list_test=[]
states_list_test=[]
@jit
# with Eloc_i, used for RGN
def sample_more_test(inputs, i):
    inputs, _ = jax.lax.scan(update_test, inputs, None, d)
    key, states,  fftW0, b0 = inputs
    logxstates, dlogx_dweights = jcompute(states, fftW0, b0)
    Hxstates = jqueryHx_x(
        states, fftW0, b0, logxstates)
    return (key, states, fftW0, b0), (states, logxstates, dlogx_dweights, Hxstates,)


@jit
def update_test(inputs, i):
    key, states, fftW0, b0 = inputs
    key, key1, key2 = random.split(key, num=3)
#     xstates=jansatzcompute_less(states,W0,b0)
    i0 = jnp.arange(parallel)
    perturbs = states

    select = random.uniform(key1, shape=(parallel,))*(d // 2)
    ups = jnp.cumsum(perturbs, axis=-1)
    i1 = jnp.argmax(jnp.greater(ups, select[:, jnp.newaxis]), axis=-1)
    select = random.uniform(key2, shape=(parallel,))*(d // 2)
    downs = jnp.cumsum(~perturbs, axis=-1)
    i2 = jnp.argmax(jnp.greater(downs, select[:, jnp.newaxis]), axis=-1)

    perturbs = perturbs.at[i0, i1] .set(~perturbs[i0, i1])
    perturbs = perturbs.at[i0, i2] .set(~perturbs[i0, i2])

    flips = (2*jnp.sum(perturbs, axis=-1) + perturbs[..., 0] > d)
    perturbs = perturbs ^ jnp.expand_dims(flips, -1)

    logxperturbs = jlogansatz(perturbs, fftW0, b0)
    logxstates = jlogansatz(states, fftW0, b0)

    flipornot = random.exponential(key, (parallel,))
    flipornot = (-0.5*flipornot < logxperturbs.real-logxstates.real)

    states = jax.numpy.where(jnp.reshape(flipornot, (-1, 1)), perturbs, states)
    return (key, states, fftW0, b0), flipornot


for iteration in range(iterations_test):
    inputs = (key, states, fftW0, b0)
    (key, states, __, __), (store_states, store_logxstates, store_dlogx_dweights, Hloc_raw,
                            ) = jax.lax.scan(sample_more_test, inputs, None, T)
    Hloc = Hloc_raw/d-E0
    energy=Hloc.real.mean()
    E_list_test.append(energy)
    states_list_test.append(store_states)
    Hloc_list_test.append(Hloc)
    if iteration % 100 == 0:
        print(f"{iteration}  Test energy={energy:12.6f}")
        np.save(save_dir+'E_list_test', E_list_test)
        np.save(save_dir+'states_list_test', states_list_test)
        np.save(save_dir+"Hloc_list_test",Hloc_list_test)
print(f"end testing, saved E_list_test, states_list_test, Hloc_list_test in the save_dir {save_dir}")
