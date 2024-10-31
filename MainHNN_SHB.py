# Script for the implementation of an input driven Hopfield model with 
# either only short term synaptic plasticity or both short and long
# term synaptic components

import numpy as np
import matplotlib.pyplot as plt
import os

from HNN_Gen import HNN
from Eul_May import EM
from HNPlot import PlotOverlap
from keras.datasets import mnist

params = {'ytick.labelsize': 35,
          'xtick.labelsize': 35,
          'axes.labelsize' : 30,
          'font.size' : 30}
plt.rcParams.update(params)


os.system('cls')

# Define the perturbation around one of the memories
eps = 0.5
# Defines the scalar amplitude of the perturbations
sigma = 0

# Definition of the integration interval
t_ini = 0
t_end = 10
# Definition of the time step
dt = 0.01
# Probability of activation
pp = 0.2

# Generation of the Hopfield model
HN = HNN(pp, eps)
# Generation of the memories
# D = 'I' --> independent vectors
# D = 'NI' --> dependent vectors
D = 'I'
HN.net(D)

# Generation of the initial condition
HN.y0_gen()

# Generation of the single input window
x = np.arange(start=0,stop=t_end,step=dt)
T = np.size(x)

# Definition of the input
u_tot = np.zeros((HN.N,T))
#u_tot[:,300:315] = np.diag(HN.mems[:,1])@np.ones((HN.N,15)) 

# Definition of the entire solution vector
Y = np.zeros((3,T))


HN.y0_gen()
# Definition of the activation function
# ac = 's' --> differentiable sigmoid
# ac = 't' --> rectified tanh
ac = 's'
# Definition of the sign of gamma
# gam = 'P' --> Positive gamma
# gam = 'N' --> Negative gamma
gam = 'N'


for j in range(HN.P):
    buff = np.multiply(HN.mems[:,j], np.random.choice([0,1], size=(HN.N,), p=[0.4, 0.6]))
    # Generation of the Euler-Mayorama Integrator for the SDE
    EMa = EM(HN, t_ini, t_end, dt, buff, D, gam)
    # Integration of the system over the time interval
    EMa.Eu_Ma_Test(HN, sigma, u_tot, ac)

    for v in range(T):
        Y[0,v] += (1/HN.P)*np.dot(EMa.y[:,v],HN.mems[:,j])/(HN.N*HN.p)
        for h in range(HN.P):
            if h!=j:
                Y[1,v] += (1/(HN.P*(HN.P-1)))*np.dot(EMa.y[:,v],HN.mems[:,h])/(HN.N*HN.p)

for j in range(np.shape(HN.IC)[1]):
    buff = np.multiply(HN.IC[:,j], np.random.choice([0,1], size=(HN.N,), p=[0.3, 0.7]))
    # Generation of the Euler-Mayorama Integrator for the SDE
    EMa = EM(HN, t_ini, t_end, dt, buff, D, gam)
    # Integration of the system over the time interval
    EMa.Eu_Ma_Test(HN, sigma, u_tot, ac)

    for v in range(T):
        Y[2,v] += (1/np.shape(HN.IC)[1])*np.dot(EMa.y[:,v],HN.IC[:,j])/(HN.N*HN.p)


# Plotting of the overlap during training
title = 'AverageOverlap.pdf'
PlotOverlap(Y, t_end, title)

# Definition of the entire solution vector
Y1 = np.zeros((3,T))

buff = 0.7*np.ones((HN.N,))+0.001*np.random.randn(HN.N)
# Generation of the Euler-Mayorama Integrator for the SDE
EMa = EM(HN, t_ini, t_end, dt, buff, D, gam)
# Integration of the system over the time interval
EMa.Eu_Ma_Test(HN, sigma, u_tot, ac)

for v in range(T):
    Y1[0,v] = np.dot(np.ones((HN.N)),EMa.z[:,v])/HN.N

buff_y = np.zeros((T,))
for j in range(HN.P):
    for v in range(T):
        buff_y[v] = np.dot(EMa.z[:,v],HN.mems[:,j])/(HN.N)
        
    if np.sum(buff_y)>np.sum(Y[1,:]):
        Y1[1,:] = np.copy(np.transpose(buff_y))

for j in range(np.shape(HN.IC)[1]):
    for v in range(T):
        Y[2,v] += (1/np.shape(HN.IC)[1])*np.dot(EMa.y[:,v],HN.IC[:,j])/(HN.N*HN.p)


title = 'OneMemory'
PlotOverlap(Y1, t_end, title) 
