# Code for the evaluation of the stability of the model based on the parameters 
# rho, x^{\star}
# The phase diagram requires the choice \gamma>0 or \gamma<0

# Paper: "Firing Rate Models as Associative Memory: Excitatory-Inhibitory Balance for Robust Retrieval"
# Code author: Simone Betteti
# Year: 2024 

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm

params = {'ytick.labelsize': 25,
          'xtick.labelsize': 25,
          'axes.labelsize' : 30,
          'axes.titlesize' : 25,
          'legend.fontsize':30,
          'font.weight':'bold',
          'font.size':25}
plt.rcParams.update(params)

# Functions for the construction of the synaptic matrix

# Definition of the memories (random)
def Memories(N, P, q):

    mems = np.random.choice([0, 1], (N,P), p=[1-q, q])

    return mems

# Definition of the memories (deterministic)
def Memories_det(q, N):

    P = int(np.floor(1+1/q))

    mems = np.zeros((N,P))

    cor = int(q*q*N)
    it = int(np.floor(q*(1-q)*N))

    mems[:cor,:]=np.ones((cor,P))

    for i in range(it):
        mems[cor+i*P:cor+(i+1)*P,:] = np.eye(P)

    return P, mems

# Definition of the parameters for the construction of the synaptic matrix W
def W_para(a, b, phia, phib, q):

    alpha = ((b-a)/(phib-phia))

    gamma = (q*b+(1-q)*a)/(q*phib+(1-q)*phia)

    return alpha, gamma

# Definition of the synaptic matrix
def W_gen(mems, N, alpha, gamma, q):

    W = alpha*(mems-q)@np.transpose(mems-q)/(N*q*(1-q)) + gamma*np.outer(np.ones(N,),np.ones(N,))/N

    return W

# Functions for the definition of the activation functions and their derivatives

# Function for the logistic sigmoid actvation
def Sigmoid(y, rho, xstar):

    phi = 1/(1+np.exp(-4*rho*(y-xstar-(1/(2*rho)))))

    return phi

# Function for the derivative of the logistic sigmoid actvation
def DSigmoid(y, rho, xstar):

    dphi = 4*rho*np.exp(-4*rho*(y-xstar-(1/(2*rho))))/(1+np.exp(-4*rho*(y-xstar-(1/(2*rho)))))**(2)

    return dphi

# Function for the rectified hyperbolic tangent
def SatTanh(y, rho, xstar):
    if np.size(y)>1:
        # Definition of the indices to zero
        idx = np.nonzero(y < xstar)
        # Definition of the activation function
        phi = np.tanh(rho*(y-xstar))
        phi[idx] = 0
    else:
        if y<xstar:
            phi = 0
        else:
            phi = np.tanh(rho*(y-xstar))

    return phi

# Function for the derivative of the rectified hyperbolic tangent
def DSatTanh(y, rho, xstar):
    if np.size(y)>1:
        # Definition of the indices to zero
        idx = np.nonzero(y < xstar)
        # Definition of the activation function
        dphi = rho*(1-np.tanh(rho*(y-xstar))**(2))
        dphi[idx] = 0
    else:
        if y<xstar:
            dphi = 0
        else:
            dphi = rho*(1-np.tanh(rho*(y-xstar))**(2))

    return dphi

# Start of the main code

# Definition of the population size
N = 1000
# Definition of the number of memories
P = 6
# Definition of the parameters x0 and x1
# Definition of the sign of gamma: 'N' --> negative; 'P' --> positive
g = 'N'
if g == 'N':
    # Negative gamma
    a = -0.3; b = 0.95
elif g == 'P': 
    # Positive gamma
    a = 0.1; b = 0.95

# Mean Activation
q = 0.2

# Definition of the memories
# random memories
#mems = Memories(N, P, q)
# deterministic memories
P, mems = Memories_det(q, N)

# Definition of the type of activation function: 'S' -> sigmoid; 'T' -> rectified tanh
act = 'T' 

# meshsize
mx = 30
# Definition of the mesh for the parameters rho and xstar
rho = np.linspace(start = .2, stop = 5, num = mx)
xstar = np.linspace(start = 0, stop = 0.9, num = mx)

# Definition of the matrix for the phase diagram
Q = np.zeros((mx,mx))

# Definition of the matrix for the stability condition
S = np.zeros((mx,mx))
# Definition of the matrices for the instability conditions
S_ins_2 = np.zeros((mx,mx))

for i in tqdm(range(mx)):
    for j in range(mx):
        # arrays for the mean of the eigenvalues
        eig_max = np.zeros((P,))
        eig_min = np.zeros((P,))
        # Definition of the values y0 and y1
        if act == 'S':
            phia = Sigmoid(a, rho[i], xstar[j]); phib = Sigmoid(b, rho[i], xstar[j])
            da = DSigmoid(a, rho[i], xstar[j]); db = DSigmoid(b, rho[i], xstar[j])
        elif act == 'T':
            phia = SatTanh(a, rho[i], xstar[j]); phib = SatTanh(b, rho[i], xstar[j])
            da = DSatTanh(a, rho[i], xstar[j]); db = DSatTanh(b, rho[i], xstar[j])

        # Definition of the maximal derivative
        eta = np.max([da, db])
        eta_min = np.min([da,db])

        # Definition of the synaptic parameters
        alpha, gamma = W_para(a, b, phia, phib, q)

        # Definition of the maximal synaptic parameter
        spar = np.max([alpha, gamma])

        # Sufficient stability condition
        if eta*spar >= 1:
            S[i,j] = 1

        # instability factor 
        #inst = np.max([da*(q*alpha+(1-q)*gamma),db*((1-q)*alpha+q*gamma)])
        inst_alt = np.max([q*(gamma-alpha)*(db-da)-(1-db*alpha)*(1-da*gamma),q*db*((1-q)*alpha+q*gamma)-1])
        # Instability condition
        if inst_alt > 0:
            S_ins_2[i,j] = 1

        # Definition of the synaptic matrix
        W = W_gen(mems, N, alpha, gamma, q)

        # Computing the eigenvalues of the jacobian of the firing rate field for each of the memories
        for p in range(P):
            # Computation of the internal field
            I = (b-a)*mems[:,p]+a*np.ones((N,))
            # Computation of the derivative
            if act == 'S':
                Dpart = DSigmoid(I, rho[i], xstar[j])
                
            elif act == 'T':
                Dpart = DSatTanh(I, rho[i], xstar[j])
                

            # Compuation of the Jacobian
            J = -np.eye(N)+np.sqrt(np.diag(Dpart))@W@np.sqrt(np.diag(Dpart))

            eig, _ = np.linalg.eigh(J)

            eig_max[p] = np.max(eig)
            eig_min[p] = np.min(eig)

        # Computation of the average maximum and minimum eigen values for the Jacobian
        avgeig_max = np.max(eig_max)
        avgeig_min = np.min(eig_max)

        # Setting the entry of the matrix on the base of the stability properties
        if avgeig_max<0:
            pass
        elif avgeig_max>=0:
            Q[i,j] = 1


# Visualization with the countour plot
[Y,X] = np.meshgrid(xstar,rho)
plt.figure(figsize=(15,15))
plt.contourf(Y, X, Q, levels = [-1, 0, 1], colors = ['lightskyblue', 'lightsalmon'], alpha=.7)
plt.contourf(Y, X, S, levels = [-1, 0, 1], colors = ['navy', 'red'], alpha=[.5, 0])
plt.contourf(Y, X, S_ins_2, levels = [-1, 0, 1], colors = ['navy', 'firebrick'], alpha=[0, .5])
plt.scatter(0.2, 4.8, s=650, color='darkblue', marker='s', linewidth=2.0, edgecolor='black')
plt.scatter(0.8, 4.8, s=800, color='maroon', marker='*', linewidth=2.0, edgecolor='black')
plt.plot(np.nan, np.nan, color = 'firebrick', label = 'Instability condition', linewidth=6.0)
plt.plot(np.nan, np.nan, color = 'salmon', label = 'Numerical instability', linewidth=6.0)
plt.plot(np.nan, np.nan, color = 'lightskyblue', label = 'Numerical stability', linewidth=6.0)
plt.plot(np.nan, np.nan, color = 'navy', label = 'Stability condition', linewidth=6.0)
plt.legend()
plt.contour(Y, X, Q, levels = [0, 1], colors=['lightskyblue'], linewidths=4.0)
plt.contour(Y, X, S, levels = [0, 1], colors=['navy'] ,linestyles='dashed', linewidths=4.0)
plt.contour(Y, X, S_ins_2, levels = [0, 1], colors=['maroon'] ,linestyles='dotted', linewidths=4.0)
plt.ylabel(r'$\rho$')
plt.xlabel(r'$\mathrm{I}^{\star}$')
plt.savefig("PhaseLevelDiagram1.pdf", bbox_inches = 'tight', format = 'pdf')
plt.close()

