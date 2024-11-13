# Script for the computation of the Energy function associated to different firing rate models

# Paper: "Firing Rate Models as Associative Memory: Excitatory-Inhibitory Balance for Robust Retrieval"
# Code author: Simone Betteti
# Year: 2024 

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

params = {'ytick.labelsize': 10,
          'xtick.labelsize': 10,
          'axes.labelsize' : 30,
          'axes.titlesize' : 30,
          'font.weight':'bold',
          'font.size':25}
plt.rcParams.update(params)

# Definition of the inverse function(s) and integrals

# Inverse and integral of the sigmoid activation function
def Sigmoid(x,sl, xstar):
    # Definition of the inverse functions
    inv_phi = -(1/sl)*np.log((1/x)-1)+xstar+1/(2*sl)

    # Definition of the integral
    inte = (1/sl)*(np.log(1+np.exp(sl*(inv_phi-xstar-1/(2*sl))))-np.log(1+np.exp(-sl*(xstar+1/(2*sl)))))

    return inv_phi, inte

# Sigmoid activation function
def SigmoidF(x, sl, xstar):
    # definition of the activation
    phi = 1/(1+np.exp(-sl*(x-xstar-1/(2*sl))))

    return phi

# Inverse and integral of the rectified tanh activation function
def SatTanh(x,sl):
    # Definition of the saturation point
    x_star = 0.2
    # Definition of the inverse function
    inv_phi = np.log((1+x)/(1-x))/(2*sl)+x_star

    # Definition of the integral
    inte = np.log(np.cosh(sl*(inv_phi-x_star)))/sl

    return inv_phi, inte

# Rectified tanh activation function
def SatTanhF(x,sl):
    # Definition of the saturation point
    x_star = 0.2
    if np.size(x)>1:
        # Definition of the indices to zero
        idx = np.nonzero(x < x_star)
        # Definition of the activation function
        phi = np.tanh(sl*(x-x_star))
        phi[idx] = 0
    else:
        if x<x_star:
            phi = 0
        else:
            phi = np.tanh(sl*(x-x_star))

    return phi

# Function for the generation of the parameters necessary for the creation of the synaptic matrix
def SynMatP(a, b, phia, phib):
    # Definition of the parameter alpha
    alpha = (b-a)/((1/5)*(1-1/5)*(phib-phia))
    # Definition of the beta parameter
    beta = 1/5
    # Definition of the gamma parameter
    gamma = ((1-beta)*a+beta*b)/((1-(1/5))*phia+(1/5)*phib)

    return alpha, beta, gamma

# Definition of the scaled synaptic matrix
def SynMat(alpha, beta, gamma, N):
    # Definition of the bidimensional memory vectors
    P = np.int(np.floor(N/(6*np.log(N))))
    mems = np.random.choice(a=[0, 1], size = (N, P), p = [0.8, 0.2])



    # Definition of the scaled synaptic matrix
    W = (alpha/N)*(mems-beta)@np.transpose(mems-beta)+gamma*np.ones((N,N))/N

    return W, mems

# Definition of the network size
N = 1024
# Definition of the slope for the sigmoid
sl = 4.8*4
# Definition of the necessary parameters
xstar = 0.2
# Definition of the memory values
#Positive gamma
#a = 0.1; b = 0.95
# Negative gamma 
a = -0.3; b = 0.95
# Definition of the respective activation values
phia = SigmoidF(a, sl, xstar); phib = SigmoidF(b, sl, xstar)
#phia = SatTanhF(a, sl); phib = SatTanhF(b, sl)

# Define the difference with the boundary
boundary_diff = 1 - phib


# Definition of the parameters for the construction of the synaptic matrix
alpha, beta, gamma = SynMatP(a, b, phia, phib)
print(gamma)

# Definition of synaptic matrix
W, mems = SynMat(alpha, beta, gamma, N)

# Definition of the computation mesh
x = np.linspace(start=0.01, stop=phib+boundary_diff, num=50)
[X, Y] = np.meshgrid(x,x)

# Definition of the vector field mesh
x1 = np.linspace(start=0.01, stop=phib+boundary_diff, num=15)
[X1, Y1] = np.meshgrid(x1,x1)

# Definition of the Energy matrix
E = np.zeros(np.shape(X))

# Definition of the memories for the plane
v1 = (phib-phia)*mems[:,0]+phia*np.ones((N,))
v2 = (phib-phia)*mems[:,1]+phia*np.ones((N,))

# Computation of the Energy values
for h in range(np.size(x)):
    for k in range(np.size(x)):
        # Definition of useful values
        inv_phi = np.zeros((N,))
        inte = np.zeros((N,))

        # Definition of the point on the plane in which the energy is evaluated
        v = x[h]*v1 + x[k]*v2
        NoEn = np.count_nonzero(v>=1)
        if NoEn>0:
            E[h,k] = 10000
        else:
            for i in range(N):
                #inv_bf, int_bf = SatTanh(v[i], sl)
                inv_bf, int_bf = Sigmoid(v[i], sl, xstar)

                inv_phi[i] = inv_bf
                inte[i] = int_bf

            # Computation of point-wise energy value
            E[h,k] = -(1/2)*np.dot(W@v,v)+np.dot(inv_phi,v)-np.sum(inte)


E_buff = np.copy(E)
E_buff[E==10000] = -10000
# Plotting values for the colormap
vma = np.max(E_buff)
vmi = np.min(E)

E[E==10000] = np.nan

txt_en = 'EnergyFiringHD.pdf'

fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(projection='3d')
ax1.set_zlim(vmi-5,vma+0.1)
ax1.plot_surface(X,Y,E, cmap=matplotlib.cm.hot, vmin=vmi-2, vmax=vma+5, lw=0)
ax1.contour(X,Y,E, 40, linewidths=1, cmap="gray", linestyles="solid", offset=vmi-5)
ax1.scatter(1, 0, vmi-5, color='green', linewidth=7.0)
ax1.text(0.85, -0.02, vmi-5, s=r'$\bar \xi^{1}$', color='black')
ax1.scatter(0.01, 0.01, vmi-5, color='purple', linewidth=8.0)
ax1.text(0.01+0.07, 0.01-0.08, vmi-5, s=r'$\vec{0}$', color='black')
ax1.set_xlabel(r'$t_{1}$')
ax1.set_ylabel(r'$t_{2}$')
ax1.set_title(r'$\text{E}_{FR}(x)$')
ax1.grid(False)
plt.savefig(txt_en,bbox_inches='tight',format='pdf')
plt.close()

E_csv = pd.DataFrame(E)
E_csv.to_csv("Energy.csv", index=False)

X_csv = pd.DataFrame(X)
X_csv.to_csv("Mesh.csv", index=False)
