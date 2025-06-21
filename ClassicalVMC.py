#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:
# update: this file is named improve because we do not save ansatz, but rather save log(ansatz) to avoid overflow.

# the version which need enough storage of large matrices. On the contrary, 1dxxz is update and accumulate on
# first and second moments directly, without higher requirement of storage capacity, but with the price that we
# cannot make use of the parallel advantage of GPU multiplying large matrices. So although 1dxxz_store and 1dxxz
# 's performance is close on CPU, 1dxxz_store is 3 times faster than 1dxxz when they are both run on GPU.

# 1d xxz model, finally works, note: we are updating ansatz based on sampling in subspace, so the final features
# and bias only apply to the values of sites in the subspace. The values outside the subspace are not guaranteed to
# be correct. So be careful if you want to calculate rayleigh quotient.

# 11/30/2021 this is the initail version of generating E_list using vanilla sampler. use E[(X-E[X](Y-E(Y)))] to
# compute covariances in S,g and H. That is different from what we use in improve_original_exy version, which instead
# use E[XY]-E[X]E[Y]. The instability issue is not changed. The stability of this one is a bit better (energy spikes
# shows later than that one)

from tempfile import TemporaryFile
import pathlib
import os
import math
import sys
import copy
import scipy
import pickle

from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import spsolve

import jax.numpy as jnp
import numpy as np
import jax.random as random
from jax import pmap
from jax import vmap
from jax import jit
import jax.ops
import time
from jax import config
config.update("jax_enable_x64", True)


@jit
def logansatz(s, fftW0, b0):
    Fsigma = jnp.fft.fft(2*s-1)
    theta = jnp.fft.ifft(fftW0*jnp.conj(Fsigma))+b0
    logx_sigma = jnp.sum(jnp.log(jnp.cosh(theta)))
#     x_sigma=jnp.exp(logx_sigma)
    return logx_sigma


logansatz1 = vmap(logansatz, (0, None, None), 0)
jlogansatz = jit(logansatz1)


@jit
def compute(s, fftW0, b0):
    Fsigma = jnp.fft.fft(2*s-1)
    theta = jnp.fft.ifft(fftW0*jnp.conj(Fsigma))+b0
    logx_sigma = jnp.sum(jnp.log(jnp.cosh(theta)))
    tanhtheta = jnp.tanh(theta)
    dlogx_sigma_db = jnp.sum(tanhtheta, axis=-1)
    dlogx_sigma_dw = jnp.fft.ifft(jnp.fft.fft(tanhtheta)*Fsigma)
    dlogx_dweights = jnp.concatenate(
        (dlogx_sigma_dw.flatten(), jnp.reshape(dlogx_sigma_db, (-1,))))
    return logx_sigma, dlogx_dweights


compute1 = vmap(compute, (0, None, None), 0)
jcompute = jit(compute1)


@jit
# with Eloc_i, used for RGN
def sample_more(inputs, i):
    inputs, _ = jax.lax.scan(update, inputs, None, d)
    key, states,  fftW0, b0 = inputs
    logxstates, dlogx_dweights = jcompute(states, fftW0, b0)
    Hxstates = jqueryHx_x(
        states, fftW0, b0, logxstates)
    return (key, states,  fftW0, b0), (states, logxstates, dlogx_dweights, Hxstates, )


@jit
def update(inputs, i):
    key, states,  fftW0, b0 = inputs
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
    perturbs = perturbs.at[i0, i1].set(~perturbs[i0, i1])
    perturbs = perturbs.at[i0, i2].set(~perturbs[i0, i2])

    flips = (2*jnp.sum(perturbs, axis=-1) + perturbs[..., 0] > d)
    perturbs = perturbs ^ jnp.expand_dims(flips, -1)

    logxperturbs = jlogansatz(perturbs, fftW0, b0)
    logxstates = jlogansatz(states, fftW0, b0)

    flipornot = random.exponential(key, (parallel,))
    flipornot = (-0.5*flipornot < logxperturbs.real-logxstates.real)

    states = jax.numpy.where(jnp.reshape(flipornot, (-1, 1)), perturbs, states)
    return (key, states,  fftW0, b0), flipornot


@jit
def queryHx_x(state, fftW0, b0, logxstate):
    sames = state ^ state[(jnp.arange(d) + 1) % d]
    energy1 = 2*jnp.sum(sames)-d
    Hx1_x = -delta*energy1

    statesxy = jnp.repeat(jnp.expand_dims(state, -1), d, -1).T
    i0 = jnp.arange(d)
    statesxy = statesxy.at[i0, i0].set(~statesxy[i0,  i0])
    statesxy = statesxy.at[i0, (i0+1) % d].set(~statesxy[i0,  (i0+1) % d])
    # sames=~sames

    flips = (2*jnp.sum(statesxy, axis=-1) + statesxy[..., 0] > d)
    statesxy = statesxy ^ jnp.expand_dims(flips, -1)

    logxstatesxy, dlogxstatesxy = jcompute(statesxy, fftW0, b0)
    wave_ratios = jnp.exp(logxstatesxy-logxstate)
    sameswave_ratios = sames*wave_ratios
    Hx2_x = -2*jnp.sum(sameswave_ratios)
    return Hx1_x+Hx2_x


queryHx_x1 = vmap(queryHx_x, (0, None, None, 0), 0)
jqueryHx_x = jit(queryHx_x1)


@jit
def queryHx_x_more(state, fftW0, b0, logxstate, dlogxstate_dweights):
    sames = state ^ state[(jnp.arange(d) + 1) % d]
    energy1 = 2*jnp.sum(sames)-d
    Hx1_x = -delta*energy1
    Hdx1_x = -delta*energy1*dlogxstate_dweights

    statesxy = jnp.repeat(jnp.expand_dims(state, -1), d, -1).T
    i0 = jnp.arange(d)
    statesxy = statesxy.at[i0, i0].set(~statesxy[i0,  i0])
    statesxy = statesxy.at[i0, (i0+1) % d].set(~statesxy[i0,  (i0+1) % d])
    # sames=~sames

    flips = (2*jnp.sum(statesxy, axis=-1) + statesxy[..., 0] > d)
    statesxy = statesxy ^ jnp.expand_dims(flips, -1)

    logxstatesxy, dlogxstatesxy = jcompute(statesxy, fftW0, b0)
    wave_ratios = jnp.exp(logxstatesxy-logxstate)
    sameswave_ratios = sames*wave_ratios
    Hx2_x = -2*jnp.sum(sameswave_ratios)
    Hdx2_x = -2*jnp.sum(dlogxstatesxy*sameswave_ratios.reshape(-1, 1), axis=0)
    return Hx1_x+Hx2_x, Hdx1_x+Hdx2_x


queryHx_x_more1 = vmap(queryHx_x_more, (0, None, None, 0, 0), 0)
jqueryHx_x_more = jit(queryHx_x_more1)


@jit
def cvcompute(state):
    sames = state ^ state[(jnp.arange(d) + 1) % d]
    energy1 = 2*jnp.sum(sames)-d
    return -energy1/d


cvcompute1 = vmap(cvcompute, 0, 0)
jcvcompute = jit(cvcompute1)


def printparameters():
    print("d=", d)
    print("delta=", delta)
    print("E0=", E0)
    print("parallel=", parallel)
    print("myseed=", myseed)
    print("T=", T)
    print("eta=", eta)
    print("epsilon=", epsilon)
    print("alpha=", alpha)
    print("iterations=", iterations)
    print("iterations0=", iterations0)


eta = 0.01
epsilon = 0.01
T = 20
setting = sys.argv
d = int(setting[1])
delta = float(setting[2])
parallel = int(setting[3])
myseed = int(setting[4])
alpha = int(setting[5])
iterations = int(setting[6])

if d == 100:
    if delta == 0.5:
        E0 = -1.5002721151896314
    elif delta == 1.0:
        E0 = -1.772918283003674
    elif delta == 1.5:
        E0 = -2.093819885253906
    else:
        print("no defined E0")
if d == 200:
    if delta == 0.5:
        E0 = -1.5000680204043535
    elif delta == 1.0:
        E0 = -1.772671065044832
    elif delta == 1.5:
        E0 = -2.093740234375
    else:
        print("no defined E0")


key = random.PRNGKey(myseed)
E_list = []
g_list = []
epsilon_record = []
rgn_min = .001
rgn_max = 1000
reg_min = .001
reg_max = .1

iterations0 = 1000
guidance = 0.1
sr_min = 0.001
sr_max = 0.01
sr_reg = 0.001

printparameters()
epsilon_list = sr_min*(sr_max/sr_min)**np.arange(2 *
                                                 iterations/iterations0, step=2/iterations0)
epsilon_list[epsilon_list > sr_max] = sr_max
epsilons_reset = np.copy(epsilon_list)
regs = reg_min*(reg_max/reg_min)**np.arange(2 *
                                            iterations/iterations0, step=2/iterations0)
regs[regs > reg_max] = reg_max

key, key1, key2 = random.split(key, num=3)
weights_save = .001*random.normal(key1, shape=(alpha*(d + 1),)
                                  ) + .001j*random.normal(key2, shape=(alpha*(d + 1),))

weights = jnp.asarray(weights_save)

states_old = jnp.tile(jnp.array([True, False]), d//2)

key3 = random.split(key, num=parallel)
states_save = vmap(random.permutation, in_axes=(
    0, None), out_axes=0)(key3, states_old)
flips = (2*jnp.sum(states_save, axis=-1) + states_save[..., 0] > d)
states_save = states_save ^ jnp.expand_dims(flips, -1)
states_save = jnp.reshape(states_save, (parallel, -1))

states = jnp.array(states_save)

# states=random.choice(key,jnp.asarray([True, False]),(parallel,d))
W0 = jnp.reshape(weights[:-alpha], (alpha, d))
b0 = jnp.reshape(weights[-alpha:], (alpha, 1))
fftW0 = jnp.fft.fft(W0)

# In[11]:

diff_list = []
Hloc_list = []
move_list = []
states_list = []
g_list = []
S_list = []
H_list = []
H1_list = []
H2_list = []
H3_list = []
regular_list = []
w_list = []
paccfliplist = []
logx_list = []

# create the folder and set the path
timestr = time.strftime("%Y%m%d-%H%M%S")
pathname = str("./SRXXZclean1_0/")
pathlib.Path(pathname).mkdir(parents=True, exist_ok=True)
# create the readme file recording parameters
with open(pathname+'test.txt', 'w') as f:
    f.write('delta='+str(delta)+'\n')
    f.write('d='+str(d)+'\n')
    f.write('T='+str(T)+'\n')
    f.write('time='+time.strftime("%Y%m%d-%H%M%S"))

time_sum = time.time()
time_sampling = 0.0
time_linsys = 0.0

for iteration in range(iterations):
    w_list.append(weights)
    W0 = jnp.reshape(weights[:-alpha], (alpha, d))
    fftW0 = jnp.fft.fft(W0)
    b0 = jnp.reshape(weights[-alpha:], (alpha, 1))

    inputs = (key, states,  fftW0, b0)

    start_sampling = time.time()
    (key, states,  __, __), (store_states, store_logxstates,
                             store_dlogx_dweights, Hloc) = jax.lax.scan(sample_more, inputs, None, T)

    time_sampling = time_sampling+time.time()-start_sampling

    start_linsys = time.time()
    Hloc = jnp.reshape(Hloc, (-1, 1))
    epsilon = epsilon_list[iteration]

    Ehat = jnp.mean(Hloc)
    Hloc = Hloc-Ehat

    store_dlogx_dweights = jnp.reshape(store_dlogx_dweights, (-1, alpha*(d+1)))

    vmean = jnp.average(store_dlogx_dweights, axis=0)
    store_dlogx_dweights = store_dlogx_dweights-vmean
    g = jnp.mean(store_dlogx_dweights.conj()*Hloc, axis=0)

    S = 1/T/parallel * \
        jnp.matmul(store_dlogx_dweights.conj().T, store_dlogx_dweights)
    regular = (S + sr_reg*jnp.eye(g.size))
    move = -jnp.linalg.solve(regular, g)
    # move_norm = jnp.sum(jnp.abs(move)**2)**.5

    time_linsys = time_linsys+time.time()-start_linsys

    E_list.append(Ehat.real/d)

    if (iteration % 50 == 0):
        print("%5d %.6f %.6f %.10f %.10f \n" % (iteration,
              epsilon, guidance, Ehat.real/d, Ehat.real/d-E0))
#     if iteration % 100 == 0:
#         inputs_long = (key, states, fftW0, b0, weights,
#                        myseed, epsilon_list, iteration, guidance)
#         pickle.dump(inputs_long, open(pathname+'inputs%d_%d.dump' %
#                    (myseed, iteration), 'wb'))

    if (iteration % 100 == 0):
        states_list.append(store_states[-1])
    weights = weights+move*epsilon

time_sum = time.time()-time_sum
time_rest = time_sum-time_sampling-time_linsys
print(f"time_sum={time_sum:.3f}s, time_sampling={time_sampling:.3f}s, time_linsys={
      time_linsys:.3f}s, time_rest={time_rest:.3f}s")

pickle.dump(inputs, open(pathname+'inputs%d.dump' % (myseed), 'wb'))
outfile = TemporaryFile()
np.save(pathname+'E_list%d' % (myseed), np.array(E_list)-E0)
outfile = TemporaryFile()
np.save(pathname+'states_list%d' % (myseed), np.array(states_list))
outfile = TemporaryFile()
np.save(pathname+'w_list%d' % (myseed), np.array(w_list))

