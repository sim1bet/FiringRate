# Script that, given an Hopfield Model, implements the (stochastic)
# Euler-Mayorama method of integration for the solution of a given SDE
# time [0,T]

# Paper: "Firing Rate Models as Associative Memory: Excitatory-Inhibitory Balance for Robust Retrieval"
# Code author: Simone Betteti
# Year: 2024 

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class EM:
    # Initialization of the integrator
    def __init__(self, HN, t_ini, t_end, dt, ini, D, gam):
        # definition of the initial integration time
        self.t0 = t_ini
        # definition of the final integration time
        self.T = t_end
        # definition of the step size
        self.dt = dt
        # slope of the activation function
        self.delta = 4.8
        # Tolerance for the equilibrium condition
        self.tol = 10**(-9)
        # Initial condition
        self.y0 = ini
        # memory condition
        self.D = D
        # Condition on the sign of gamma
        self.g = gam
    
    # Function that implements the Euler Mayorama method given an SDE
    def Eu_Ma_Test(self,HN, sigma, u, ac):
        
        # Perturbation
        self.sigma = sigma
        # Definition of the integration interval, appropriately partitioned
        x = np.arange(start=self.t0, stop=self.T, step=self.dt)
        # defintion of the solution vector
        self.y = np.zeros((HN.N,np.size(x)))
        # definition of the activation vector
        self.z = np.zeros((HN.N,np.size(x)))

        # Definition of the necessary quantities
        if self.g == 'N':
            a = -0.3; b = 0.95
        elif self.g == 'P':
            a = 0.1; b = 0.95

        if ac == 's':
            phia = self.g_fun_sig(a); phib = self.g_fun_sig(b)
        elif ac == 't':
            phia = self.g_fun_cut(a); phib = self.g_fun_cut(b)
        # Definition of the parameters for the synaptic matrix
        if self.D == 'I':
            r = HN.p
        elif self.D == 'NI':
            r = 0.4
        alpha, beta, gamma = self.SynMatP(a, b, phia, phib, HN.p, r)

        # setting of the initial condition
        self.y[:,0] = self.y0
        # setting of the transformed initial condition
        #self.z[:,0] = self.g_fun_sig(self.y[:,0])
        self.z[:,0] = self.g_fun_cut(self.y[:,0])
        # Solution of the system using Euler-Mayorama method
        for t in tqdm(range(np.size(x)-1)):

            # Computation of the value field
            Hf = self.hop_field_test(self.y[:,t],HN, u[:,t], alpha, beta, gamma, ac)
            # Computation of the Brownian increments
            dW = np.random.randn(np.shape(self.y[:,t])[0],)
            # Computation of the field updated value
            self.y[:,t+1] = self.y[:,t] + Hf*self.dt + self.sigma*np.sqrt(self.dt)*dW
            # Computation of the overlap for the Sigmoid activation function
            if ac == 's':
                self.z[:,t+1] = self.g_fun_sig(self.y[:, t+1])
            # Computation of the overlap for the thresholded tanh
            elif ac == 't':
                self.z[:,t+1] = self.g_fun_cut(self.y[:, t+1])


    # Function that implements the positive activation function
    def g_fun_sig(self, y):
        # Definition of the flection point
        xstar = 0.2
        z = 1/(1+np.exp(-4*self.delta*(y-xstar-1/(2*self.delta))))

        return z
    
    # Function that implements the rectified tanh
    def g_fun_cut(self, y):
        # Definition of the cut point
        xstar = 0.75
        if np.size(y)>1:
            # Indices where the rate is below threshold
            idx = np.nonzero(y<xstar)
            # Activation
            z = np.tanh(self.delta*(y-xstar))
            z[idx] = 0
        else:
            if y>xstar:
                z = np.tanh(self.delta*(y-xstar))
            else:
                z = 0

        return z
    
    # Function that implements the computation of the Hopfield-field (hehe)
    def hop_field_test(self, y, HN, u, alpha, beta, gamma, ac):
        
        # Construction of the synaptic matrix
        W = alpha/(HN.N)*(HN.mems-beta)@np.transpose(HN.mems-beta)
        #W[B,B] = 0
        W += gamma/(HN.N)*np.ones((HN.N,HN.N))
        
        # Sigmoidal activation function
        if ac == 's':
            val = -y + self.g_fun_sig((W@y + u))
        # Thresholded tanh activation function
        elif ac == 't':
            val = -y + self.g_fun_cut((W@y + u))

        return val
    

    # Function with the parameters necessary for the construction of the synaptic matrix W
    def SynMatP(self, a, b, phia, phib, q, r):
        # definition of the parameter alpha
        alpha = (b-a)/(q*(1-r)*(phib-phia))
        # definition of the parameter beta
        beta = q*((1-r)*phia+r*phib)/((1-q)*phia+q*phib)
        # definition of the parameter gamma
        gamma = ((1-beta)*a+beta*b)/((1-q)*phia+q*phib)

        return alpha, beta, gamma

