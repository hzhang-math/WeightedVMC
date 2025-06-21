import os
import math
import sys
import copy
import scipy
import pickle
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import spsolve
from jax.nn import logsumexp
from jax.nn import log_softmax 
from jax.nn import softmax
import matplotlib.pyplot as plt
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
import pathlib
from tempfile import TemporaryFile
from scipy.interpolate import interp1d
from jax.numpy.linalg import norm
import matplotlib as mpl
from matplotlib.patches import Patch

@jit
def update(inputs,i):
    (key,states,cvreplica,cvstates)=inputs
    key, key1,key2 = random.split(key, num=3)
#     xstates=jansatzcompute_less(states,W0,b0)
    i0=jnp.arange(parallel*replicas)
    perturbs=states 
    select = random.uniform(key1, shape = (parallel*replicas,))*(d // 2)
    ups = jnp.cumsum(perturbs, axis = -1)
    i1 = jnp.argmax(jnp.greater(ups, select[:, jnp.newaxis]), axis = -1)
    select = random.uniform(key2, shape = (parallel*replicas,))*(d // 2)
    downs = jnp.cumsum(~perturbs, axis = -1)
    i2 = jnp.argmax(jnp.greater(downs, select[:, jnp.newaxis]), axis = -1)
    perturbs=perturbs.at[i0, i1].set(~perturbs[i0, i1])
    perturbs=perturbs.at[i0, i2].set(~perturbs[i0, i2]) 
    flips = (2*jnp.sum(perturbs, axis = -1) + perturbs[..., 0] > d)
    perturbs = perturbs ^ jnp.expand_dims(flips, -1)

    cvperturbs=jcvcompute(perturbs)
    logxpotential=jlogpotential(cvstates, cvreplica)
    logxpotperturbs=jlogpotential(cvperturbs, cvreplica)
    flipornot=random.exponential(key,(parallel*replicas,))

    logpotdiff=(logxpotperturbs-logxpotential)*0.5    
    flipornot=(-0.5*flipornot<logpotdiff)
        
    states=jax.numpy.where(jnp.reshape(flipornot, (-1,1)),perturbs,states)
    cvstates=jax.numpy.where(flipornot,cvperturbs,cvstates) 
    return (key,states,cvreplica,cvstates), cvstates


##with Eloc_i, used for RGN
@jit
def sample_more(inputs,i):
    inputs,_=jax.lax.scan(update,inputs,None,d)
    key,states,cvcenters,cvstates=inputs
    cvstates=jcvcompute(states)
    return (key,states,cvcenters,cvstates),(states,\
             cvstates)

@jit
def cvcompute(state):
    d=state.shape[-1]
    sames= state ^ state[(jnp.arange(d) + 1) % d]
    energy1 = 2*jnp.sum(sames)-d
    return -energy1/d
cvcompute1 = vmap(cvcompute, 0, 0)
jcvcompute= jit(cvcompute1) 

@jit
def logpotential(cvstate, cvreplica,cvmin,cvmax,cvlambda):
    bool_constcv=((cvreplica==cvmin) & (cvstate< cvmin)) | ((cvreplica==cvmax) & (cvstate> cvmax))
    pot=-(cvstate-cvreplica)**2/2/cvlambda**2*(~bool_constcv)
    return pot
logpotential1 = vmap(logpotential, (0, 0, None, None, None), 0)
jlogpotential= jit(logpotential1) 

logpotential21 = vmap(logpotential, (0, None, None, None, None), 0)
jlogpotential2= jit(logpotential21) 

@jit 
def compute_stratified_margin(cvstate,Bk):
    f=((Bk[:-1]<cvstate)&(Bk[1:]>=cvstate)) 
    return f
stratified_margin = vmap(compute_stratified_margin, ( 0,None), 0)
jstratified_margin= jit(stratified_margin)
stratified_margin2 = vmap(jstratified_margin, ( 0,None), 0)
jjstratified_margin= jit(stratified_margin2)
    
@jit 
def compute_stratified_f(cvstate,logx,Bk):
    f=jnp.log((Bk[:-1]<cvstate)&(Bk[1:]>=cvstate))+(2*logx.real)
    return f
stratified_f = vmap(compute_stratified_f, (0, 0,None), 0)
jstratified_f= jit(stratified_f)
stratified_f2 = vmap(jstratified_f, (0,0,None), 0)
jjstratified_f= jit(stratified_f2)

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

def logdotexp(A, B):
    max_A = jnp.max(A)
    max_B = jnp.max(B)
    C=vmap(jnp.dot, in_axes = (0, 0),out_axes = 0)\
        (jnp.exp(A - max_A),jnp.exp(B - max_B))
    C=jnp.log(C)
    C += max_A + max_B
    return C


@jit 
def logansatz(s,fftW0,b0):# k is the s is in kth cv bin
    Fsigma=jnp.fft.fft(2*s-1)
    theta=jnp.fft.ifft(fftW0*jnp.conj(Fsigma))+b0
    logx_sigma=jnp.sum(jnp.log(jnp.cosh(theta)))
#     x_sigma=jnp.exp(logx_sigma)
    return logx_sigma
logansatz1 = vmap(logansatz, (0, None, None), 0)
jlogansatz= jit(logansatz1)
logansatz2 = vmap(jlogansatz, (0, None, None), 0)
jjlogansatz= jit(logansatz2)

logansatz3 = vmap(jjlogansatz, (None, 0, 0), 0)
jjlogansatz_linear_combination= jit(logansatz3)

def EMUS_margin_density(key,pltx, cvcenters,Bk,replicas,cvlambda,parallel,T,store_cvstates,store_states,fftW0,b0,linear_combination,V):
    cvmin=cvcenters[0]
    cvmax=cvcenters[-1]
    nk=Bk.size-1
    store_marginstates=jjstratified_margin(store_cvstates,Bk)    
    
    if linear_combination==False:
        store_logxstates= jjlogansatz(store_states,fftW0,b0) 
    else:
        store_logxstates_all =jjlogansatz_linear_combination(store_states,fftW0,b0)
        store_logxstates= logsumexp(store_logxstates_all,b=V.reshape((2,1,1)),axis=0)
    
    store_logfstates=jjstratified_f(store_cvstates,store_logxstates,Bk)
    store_cvstates=store_cvstates.transpose(1,0).reshape(replicas,-1)
    
    store_logxstates=store_logxstates.transpose(1,0).reshape(replicas,-1)

    logpsii_replica=vmap(jlogpotential2, in_axes=(0,None,None,None,None),out_axes=0)(store_cvstates,cvcenters,cvmin,cvmax,cvlambda)
    dev=1.0
    u=jnp.ones(replicas)
    while dev>1e-4:
        logweighted_psi=logpsii_replica-jnp.log(u)

        normed_weighted_psi=softmax(logweighted_psi,axis=-1)
        F=jnp.mean(normed_weighted_psi,axis=1)
        q, _ = jnp.linalg.qr(F-jnp.eye(replicas))

        w=q[:,-1]/jnp.sum(q[:,-1])
        w_in=w[None,:]
        for i in range(5):
            w_in=w_in@F
        w=w_in.squeeze() 
        unew=u*w
        unew=unew/jnp.sum(unew)
        dev=jnp.linalg.norm(unew-u)
        u=unew
        u=np.maximum(u,1e-32)
    print("u=",u)

    logsum_weighted_psi=logsumexp(logweighted_psi,axis=-1,keepdims=True)
#     sum_weighted_psi=jnp.exp(logsum_weighted_psi)
    #form matrices
    store_logfstates=store_logfstates.transpose(1,0,2).reshape(replicas,-1,nk)
#     store_marginstates=store_marginstates.transpose(1,0,2).reshape(replicas,-1,nk)

    logEi1star=logsumexp(-logsum_weighted_psi,axis=1)-np.log(parallel*T)  #nansatztemp=(8,50,1), Ei1star=pi_i (1*)
    logfmeanstar=logsumexp(store_logfstates-logsum_weighted_psi,axis=1)-np.log(parallel*T)
#     marginmeanstar=jnp.mean(store_marginstates/sum_weighted_psi,axis=1) #vmeanstar=pi_i [v_j*] 

    logw=jnp.log(w)[:,None]
    logzipi=logsumexp(logw+logEi1star)
    zipi=jnp.exp(logzipi)

    logfmean=logsumexp(logw+logfmeanstar,axis=0)-logzipi
#     marginmean=jnp.dot(w,marginmeanstar)/zipi
#     store_cv=store_cvstates.reshape(-1,)
#     store_logx=store_logxstates.reshape(-1,)
    
    x = (Bk[:-1]+Bk[1:])/2
    y1=logfmean
    f1 = interp1d(x, y1, kind='linear')    
    pltf=f1(pltx)
    return pltx,pltf

def resample(init_conf,combine_save,parallel_vmc,w_resample):
    rng = np.random.default_rng()
    combine=-(combine_save.reshape(parallel_vmc,-1).transpose())[-1]
    p_combine=softmax(combine+jnp.log(w_resample))
    dist=rng.multinomial(parallel, p_combine, size=1).squeeze()
    resampled_states = np.empty( shape=(0, d) ,dtype=bool)
    for i in range(dist.size):
        resampled_states=np.vstack((resampled_states,np.tile(init_conf[i],(dist[i],1))))
    return resampled_states

def EMUS_margin_density_wrapper(fftW0,b0,alpha=5,space="subspace",d=100,parallel=10,linear_combination=False,V=None):
    myseed=123
    if d==100:
        T=500
        cvlambda=4/d
        Bk=np.arange(-1.02,1+0.12,0.12) #function size 17
        cvcenters=np.arange(-1.,1+0.08,0.08)
        pltx=(Bk[:-1]+Bk[1:])/2
        replicas=cvcenters.size #window size 26
        cvcenters=jnp.linspace(-1,1,replicas)
        cvreplica_save=jnp.repeat(cvcenters,parallel)
        cvreplica=copy.deepcopy(cvreplica_save)
        key = random.PRNGKey(myseed)
    else:
        T=1000
        Bk=np.arange(-1-2/d,1+2/d+4/d,4/d) #function size 17
        pltx=(Bk[:-1]+Bk[1:])/2
        cvcenters=np.arange(-1.,1+2*4/d,4/d)
        cvmin=cvcenters[0]
        cvmax=cvcenters[-1]
        cvlambda=4/d
        replicas=cvcenters.size
        cvreplica_save=jnp.repeat(cvcenters,parallel)
        cvreplica=copy.deepcopy(cvreplica_save)    
        key = random.PRNGKey(myseed)
    if space=="subspace":
        test_states=np.load('/scratch/hz1994/vinla/ub_pool/ubpool_cvunif_d%d_states.npy'%d)
        test_cvstates=np.load('/scratch/hz1994/vinla/ub_pool/ubpool_cvunif_d%d_cvstates.npy'%d)
    elif space=="excitedspace":
        test_states=np.load('/scratch/hz1994/vinla/ub_pool/excitedsubspace_pool_cvunif_d%d.npy'%d)
        test_cvstates=np.load('/scratch/hz1994/vinla/ub_pool/excitedsubspace_pool_cvunif_d%d_cvstates.npy'%d) 
    elif space=="fullspace":
        test_states=np.load('/scratch/hz1994/vinla/ub_pool/fullspace_pool_cvunif_d%d.npy'%d)
        test_cvstates=np.load('/scratch/hz1994/vinla/ub_pool/fullspace_pool_cvunif_d%d_cvstates.npy'%d)    
    
    key = random.PRNGKey(0)
    print(test_states.shape,test_cvstates.shape,d)
    selected_indices =random.choice(key, test_cvstates.shape[0], shape=(100,), replace=False)
    test_states = test_states[selected_indices, :]
    test_cvstates = test_cvstates[selected_indices, :] 

    pltx,pltf=EMUS_margin_density(key,pltx,cvcenters,Bk,replicas,cvlambda,parallel,T,test_cvstates,test_states,fftW0,b0,\
                                 linear_combination,V) 
    pltf=pltf-logsumexp(pltf)
    return pltx,pltf

@jit
def queryHx_x(state, fftW0, b0,logxstate, delta):
    d=state.shape[-1]
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

queryHx_x1 = vmap(queryHx_x, (0, None, None,0,None), 0)
jqueryHx_x = jit(queryHx_x1)

def readpms(path, time,alpha,d):
    w=np.load(path+'w_list.npy')
    weights=w[time]
    W0=jnp.reshape(weights[:-alpha],(alpha,d))
    b0=jnp.reshape(weights[-alpha:],(alpha,1))
    fftW0=jnp.fft.fft(W0)
    return fftW0,b0
