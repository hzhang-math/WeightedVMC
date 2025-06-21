# %%
import wandb
from jax.scipy.special import logsumexp
from functools import partial
from jax import random, vmap
import jax
from tempfile import TemporaryFile
import pathlib
import os
import math
import sys
import copy
import scipy
import pickle
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import spsolve
import platform
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

    logxstatesxy = jlogansatz(statesxy, fftW0, b0)
    wave_ratios = jnp.exp(logxstatesxy-logxstate)
    sameswave_ratios = sames*wave_ratios
    Hx2_x = -2*jnp.sum(sameswave_ratios)
    return Hx1_x+Hx2_x


queryHx_x1 = vmap(queryHx_x, (0, None, None, 0,), 0)
jqueryHx_x = jit(queryHx_x1)


def has_gpu():
    devices = jax.devices()
    return any(device.device_kind == 'gpu' for device in devices)


def reset_len_E(store_logx):
    energies = -2*store_logx.real
    upper = np.max(energies)
    lower = np.min(energies)
    len_E = np.linspace(lower, upper, n_len_E)
    return len_E
# #
# def reset_len_E(store_logx):
#     energies = -2*store_logx.real
#     upper = np.max(energies)
#     lower = upper-100
#     len_E = np.linspace(lower, upper, n_len_E)
#     return len_E


@jit
def cvcompute(state):
    sames = state ^ state[(jnp.arange(d) + 1) % d]
    energy1 = 2*jnp.sum(sames)-d
    return -energy1/d


cvcompute1 = vmap(cvcompute, 0, 0)
jcvcompute = jit(cvcompute1)


def ind(energy, len_E):
    return np.searchsorted(len_E, energy)


ind_1 = vmap(ind, (0, None,), 0)
jind = jit(ind_1)


@jax.jit
def energy(state, fftW0, b0):
    return -2*logansatz(state, fftW0, b0).real


energy_1 = vmap(energy, (0, None, None, ), 0)
jenergy = jit(energy_1)


@partial(jax.jit)
def update_wang_landau(inputs, _):
    def step(inputs, _):
        key, fftW0, b0, states, len_E, current_energy, f, V, histogram, step = inputs
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

        new_energy = jenergy(perturbs, fftW0, b0)
        indices_current = jind(current_energy, len_E)
        indices_new = jind(new_energy, len_E)
        flipornot = random.exponential(key, (parallel,))
        flipornot = (-flipornot < V[indices_current] - V[indices_new]
                     - new_energy + current_energy)
        # flipornot = (-flipornot <
        #              - new_energy + current_energy)
        states = jax.numpy.where(jnp.reshape(
            flipornot, (-1, 1)), perturbs, states)
        current_energy = jnp.where(flipornot, new_energy, current_energy)
        indices_current = jind(current_energy, len_E)
        f = f_value*jnp.exp(-(V[indices_current])/DeltaT)
        V = V.at[indices_current].add(f)
        histogram = histogram.at[indices_current].add(1)
        return (key, fftW0, b0,  states, len_E, current_energy, f, V, histogram, step), indices_current

    inputs, indices_current = jax.lax.scan(step, inputs, None, length=d)

    key, fftW0, b0,  states, len_E, current_energy, f, V, histogram, step = inputs
    f, histogram = update_f_and_histogram(f, histogram, len_E, step)
    step += 1
    logxstates, dlogxstates = jcompute(states, fftW0, b0)
    Hxstates = jqueryHx_x(states, fftW0, b0, logxstates)
    return (key, fftW0, b0,  states, len_E, current_energy, f, V, histogram, step), (states, current_energy, f, indices_current, logxstates, dlogxstates, Hxstates)


@jax.jit
def update_f_and_histogram(f, histogram, len_E, step):
    # Check if histograms are flat independently
    mean_histogram = jnp.mean(histogram)
    min_histogram = jnp.min(histogram)

    # Condition to check if all histograms are flat
    condition = (min_histogram > flatness_criterion * mean_histogram)

    def true_fun(args):
        f, histogram = args
        # jax.debug.print("f:{}", f)
        # jax.debug.print("step:{}", step)
        print("Clear histogram")
        return f / 2, jnp.zeros(len_E.size+1)

    def false_fun(args):
        f, histogram = args
        return f, histogram

    f, histogram = jax.lax.cond(condition, true_fun, false_fun, (f, histogram))
    return f, histogram


def printparameters():
    print(f"d: {d}")
    print(f"iterations: {iterations}")
    print(f"myseed:{myseed}")
    print(f"T: {T}")
    print(f"parallel: {parallel}")
    print(f"alpha: {alpha}")
    print(f"eta: {eta}")
    print(f"epsilon: {epsilon}")
    print(f"delta: {delta}")
    print(f"n_len_E: {n_len_E}")
    print(f"batchsize_Sg: {batchsize_Sg}")
    print(f"batchsize_Hloc: {batchsize_Hloc}")
    print(f"E0: {E0}")
    print(f"f_value: {f_value}")
    print(f"DeltaT: {DeltaT}")
    print(f"savedir: {save_dir}")


setting = sys.argv
d = int(setting[1])
iterations = int(setting[2])
myseed = int(setting[3])
T = int(setting[4])
parallel = int(setting[5])
alpha = int(setting[6])
f_value = float(setting[7])
DeltaT = float(setting[8])
batchsize_Sg = int(setting[9])
batchsize_Hloc = int(setting[10])
# iterations = 1000
# T = 20
# d = 10
eta = 0.001
epsilon = 0.01
delta = 1.0
f = f_value*np.ones(parallel)
flatness_criterion = 0.8
min_f = 5e-4
n_len_E = 20
# DeltaT = 10.

# if d == 200:
#     batchsize_Sg = 2000
#     batchsize_Hloc = 500
# else:
#     batchsize_Sg = 2000
#     batchsize_Hloc = 2000

# if d == 200:
#     batchsize = 250
# else:
#     batchsize = 2000

if d == 10:
    E0 = -1.8061785417968172
elif d == 20:
    E0 = -1.780877305975289
elif d == 24:
    E0 = -1.7783357527562065
elif d == 28:
    E0 = -1.7768067916492418
elif d == 30:
    E0 = -1.7762617412219392
elif d == 36:
    E0 = -1.7751373135626913
elif d == 40:
    E0 = -1.7746522788333214
elif d == 50:
    E0 = -1.7739085293853316
elif d == 60:
    E0 = -1.7735048777082425
elif d == 70:
    E0 = -1.7732616258321707
elif d == 80:
    E0 = -1.7731038074860654
elif d == 90:
    E0 = -1.7729956386292987
elif d == 100:
    E0 = -1.772918283003674
elif d == 200:
    E0 = -1.772671065044832
else:
    print("Not found d")

key = random.PRNGKey(myseed)
key, key1, key2 = random.split(key, num=3)
weights_save = .001*random.normal(key1, shape=(alpha*(d + 1),)
                                  ) + .001j*random.normal(key2, shape=(alpha*(d + 1),))
weights = jnp.asarray(weights_save)
W0 = jnp.reshape(weights[:-alpha], (alpha, d))
fftW0 = jnp.fft.fft(W0)
b0 = jnp.reshape(weights[-alpha:], (alpha, 1))

states_old = jnp.tile(jnp.array([True, False]), d//2)
key3 = random.split(key, num=parallel)
states_save = vmap(random.permutation, in_axes=(
    0, None), out_axes=0)(key3, states_old)
flips = (2*jnp.sum(states_save, axis=-1) + states_save[..., 0] > d)
states_save = states_save ^ jnp.expand_dims(flips, -1)
states_save = jnp.reshape(states_save, (parallel, -1))
states = jnp.array(states_save)

# Functions to compute indices and flip spins
logxstates = jlogansatz(states, fftW0, b0)
len_E = reset_len_E(logxstates)
V = jnp.zeros(len_E.size+1)
histogram = jnp.zeros(len_E.size+1)

current_energy = jenergy(states, fftW0, b0)
indices = jind(current_energy, len_E)
# V = V.at[indices].add(1)
# histogram = histogram.at[indices].add(1)

# inputs = (key, states, len_E, current_energy, f,V, histogram, 0)
step = 0
w_list = []
E_list = []
histogram_list = []
len_E_list = []
store_indices_current_list = []
store_states_list = []
V_list = []
# if platform.system() == 'Linux':
#     save_dir = f"/scratch/hz1994/vinla/WL_logpsi/f_value_{f_value:.2f}/alpha_{alpha}/d_{
#         d}_T_{T}_p_{parallel}_seed_{myseed}/"
# else:
#     save_dir = f"/Users/hzhang/code_huan/WL_logpsi/f_value_{f_value:.2f}/alpha_{alpha}/d_{
#         d}_T_{T}_p_{parallel}_seed_{myseed}/"

# if platform.system() == 'Linux':
#     save_dir = f"/scratch/hz1994/vinla/welltemp_WL_logpsi/DeltaT{DeltaT}/f_value_{f_value:.2f}/alpha_{alpha}/d_{
#         d}_T_{T}_p_{parallel}_seed_{myseed}/"
# else:
#     save_dir = f"/Users/hzhang/code_huan/welltemp_WL_logpsi/DeltaT{DeltaT}/f_value_{f_value:.2f}/alpha_{alpha}/d_{
#         d}_T_{T}_p_{parallel}_seed_{myseed}/"

if platform.system() == 'Linux':
    save_dir = f"./welltemp_WL_logpsi/DeltaT{DeltaT}/f_value_{f_value:.2f}/alpha_{alpha}/d_{
        d}_T_{T}_p_{parallel}_seed_{myseed}/"
else:
    save_dir = f"/Users/hzhang/code_huan/welltemp_WL_logpsi/DeltaT{DeltaT}/f_value_{f_value:.2f}/alpha_{alpha}/d_{
        d}_T_{T}_p_{parallel}_seed_{myseed}/"



pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

printparameters()

sr_min = epsilon/10
sr_max = epsilon
iterations0 = 1000
epsilon_list = sr_min*(sr_max/sr_min)**np.arange(2 *
                                                 iterations/iterations0, step=2/iterations0)
epsilon_list[epsilon_list > sr_max] = sr_max


# @partial(jax.jit, static_argnums=(1, 5))
# def batch_queryHx_x(samples, nsamples, fftW0, b0, logsamples, batchsize):
#     Hloc = jnp.zeros((nsamples, 1)).astype(complex)
#     print("nsamples", nsamples, type(nsamples))
#     for i in range(nsamples//batchsize+1):
#         lower = i*batchsize
#         upper = (i+1)*batchsize
#         ind = jnp.arange(lower, upper)
#         batchHloc = jqueryHx_x(
#             samples[ind], fftW0, b0, logsamples[ind])[:, None]
#         Hloc = Hloc.at[ind].set(batchHloc)
#     return Hloc


@partial(jax.jit, static_argnums=(1, 2))
def computeS_batch(dlogsamples, nsamples, batchsize):
    # Initialize the S matrix
    S = jnp.zeros(
        (dlogsamples.shape[1], dlogsamples.shape[1])).astype(complex)

    # Vectorized operation to compute the contribution of all batches
    for i in range(nsamples // batchsize + 1):
        lower = i * batchsize
        upper = (i+1)*batchsize
        ind = jnp.arange(lower, upper)
        batch_dlog = dlogsamples[ind]
        S = S + jnp.matmul(batch_dlog.conj().T, batch_dlog)

    # Normalize the result
    S = S / dlogsamples.shape[0]

    return S


@partial(jax.jit, static_argnums=(2, 3))
def computeg_batch(dlogsamples, Hloc, nsamples, batchsize):
    g = jnp.zeros((dlogsamples.shape[1],)).astype(complex)
    for i in range(nsamples//batchsize+1):
        lower = i*batchsize
        upper = (i+1)*batchsize
        ind = jnp.arange(lower, upper)
        batch_dlog = dlogsamples[ind]
        batch_Hloc = Hloc[ind]
        gbatch = jnp.sum(batch_dlog.conj()*batch_Hloc, axis=0)
        g = g + gbatch
    g = g/dlogsamples.shape[0]
    return g


time0 = 0.0
time1 = 0.0
time2 = 0.0
time3 = 0.0
time4 = time.time()

wandb.login()
run = wandb.init(
    project="WL",
    config={
        "d": d,
        "alpha": alpha,
    },
)


for iteration in range(iterations):
    epsilon = epsilon_list[iteration]
    w_list.append(weights)
    W0 = jnp.reshape(weights[:-alpha], (alpha, d))
    fftW0 = jnp.fft.fft(W0)
    b0 = jnp.reshape(weights[-alpha:], (alpha, 1))

    inputs = (key, fftW0, b0,  states, len_E,
              current_energy, f, V, histogram, step)

    time_start = time.time()

    (key, fftW0, b0,  states, len_E, current_energy, f, V, histogram, step), (store_states, store_current_energy, store_f, store_indices_current, store_logx, store_dlogx_dweights, Hloc) = \
        jax.lax.scan(update_wang_landau, inputs, None, T)

    Hloc = Hloc.reshape(-1, 1)
    store_dlogx_dweights = store_dlogx_dweights.reshape(-1, alpha*(d+1))
    store_logx = store_logx.reshape(-1,)
    time_sampling = time.time()-time_start
    time0 += time_sampling

    # store_states_list.append(store_states)
    # store_indices_current_list.append(store_indices_current)
    # histogram_list.append(histogram)
    # len_E_list.append(len_E)
    # V_list.append(V)
    # print(f"histogram=", histogram)
    # formatted_len_E = np.array2string(
    #     len_E, precision=3, separator=', ')
    # print("len_E=", formatted_len_E)

    # store_states = store_states.reshape(-1, d)

    time_start = time.time()
    # store_logx, store_dlogx_dweights = jcompute(
    # store_states, fftW0, b0)
    # cvstates = jcvcompute(store_states)
    # Hloc = batch_queryHx_x(
    # store_states, T*parallel, fftW0, b0, store_logx, batchsize_Hloc).reshape(-1, 1)
    # Hloc = jqueryHx_x(store_states, fftW0, b0,
    #   store_logx).reshape(-1, 1)
    Ehat = jnp.mean(Hloc)
    Hloc = Hloc-Ehat
    if iteration % 50 == 0:
        print(f"iter = {iteration} Ehat= {(Ehat.real/d-E0):.6e} ")

    vmean = jnp.average(store_dlogx_dweights, axis=0)
    store_dlogx_dweights = store_dlogx_dweights-vmean

    time_Hloc = time.time()-time_start
    time1 += time_Hloc

    time_start = time.time()
    S = computeS_batch(store_dlogx_dweights, T*parallel, batchsize_Sg)
    grad = computeg_batch(store_dlogx_dweights, Hloc, T*parallel, batchsize_Sg)
    time_Sg = time.time()-time_start
    time2 += time_Sg

    time_start = time.time()
    # grad = jnp.mean(store_dlogx_dweights.conj()*Hloc, axis=0)
    # S = 1/T/parallel * \
    # jnp.matmul(store_dlogx_dweights.conj().T, store_dlogx_dweights)
    regular = (S + eta*jnp.eye(grad.size))
    move = -jnp.linalg.solve(regular, grad)

    time_linear = time.time()-time_start
    time3 += time_linear

    E_list.append(Ehat.real/d-E0)
    wandb.log(
        {"energy ": Ehat.real/d-E0, })
    weights = weights+move*epsilon
    if iteration % 100 == 0 or np.isnan(Ehat):
        store_states_list.append(store_states)
        np.save(save_dir+'E_list', E_list)
        np.save(save_dir+'w_list', w_list)
        np.save(
            save_dir+f'store_states_list_iter{iteration}', store_states_list)
        print(f"histogram=", histogram)
        print(f"V=", V)
        # np.save(save_dir+'histogram_list',  histogram_list)
        # np.save(save_dir+'len_E_list', len_E_list)
        # np.save(save_dir+'V_list', V_list)
        store_states_list = []

    # f = 1.0*np.zeros(parallel)
    f = f_value*np.ones(parallel)
    len_E = reset_len_E(store_logx)
    V = jnp.zeros(len_E.size+1)
    histogram = jnp.zeros(len_E.size+1)

time_total = time.time()-time4
print(f"MCMC sampling takes {time0:.3f}s,  Compute logpsi and Hloc takes{
      time1:.3f},  Form S and g takes {time2:.3f}s,   solve linear system takes {time3:.3f}s, Total time takes {time_total:.3f}s")

# %%


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


print("Starts testing")
iterations_test = 2001
T=20
states=states[:100]
print("states.shape",states.shape)
E_list_test = []
Hloc_list_test=[]
states_list_test=[]
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
