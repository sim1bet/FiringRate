# Script for the computation of the Energy function associated to different firing rate models
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

params = {'ytick.labelsize': 10,
          'xtick.labelsize': 10,
          'axes.labelsize' : 30,
          'axes.titlesize' : 25}
plt.rcParams.update(params)

# Definition of the inverse function(s) and integrals

def Sigmoid(x,sl):
    # Definition of the necessary parameters
    xstar = 0.5
    # Definition of the inverse functions
    inv_phi = -(1/sl)*np.log((1/x)-1)+xstar


    # Definition of the integral
    inte = (1/sl)*(np.log(1+np.exp(sl*(inv_phi-xstar)))-np.log(1+np.exp(-sl*xstar)))

    return inv_phi, inte

def SigmoidF(x, sl):
    # Definition of the necessary parameters
    xstar = 0.5
    # definition of the activation
    phi = 1/(1+np.exp(-sl*(x-xstar)))

    return phi

def SatTanh(x,sl):
    # Definition of the saturation point
    x_star = 0.25
    # Definition of the inverse function
    inv_phi = np.log((1+x)/(1-x))/(2*sl)+x_star

    # Definition of the integral
    inte = np.log(np.cosh(sl*(inv_phi-x_star)))/sl

    return inv_phi, inte

def SatTanhF(x,sl):
    # Definition of the saturation point
    x_star = 0.25
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
    alpha = (b-a)/((1/5)*(phib-phia))
    # Definition of the beta parameter
    beta = 1/5
    # Definition of the gamma parameter
    gamma = ((1-beta)*a+beta*b)/((1-(1/5))*phia+(1/5)*phib)

    return alpha, beta, gamma

# Definition of the scaled synaptic matrix
def SynMat(alpha, beta, gamma):
    # Definition of the bidimensional memory vectors
    xi1 = np.array([1, 0])
    xi2 = np.array([0, 1])
    on = np.ones((2,2))

    # Definition of the scaled synaptic matrix
    W = (alpha/2)*(np.outer(xi1-beta,xi1-beta)+np.outer(xi2-beta,xi2-beta))+gamma*on/2

    return W

# Definition of the slope for the sigmoid
sl = 10
# Definition of the memory values
a = -0.3; b = 0.9
# Definition of the respective activation values
#phia = SigmoidF(a, sl); phib = SigmoidF(b, sl)
phia = SatTanhF(a, sl); phib = SatTanhF(b, sl)


# Definition of the parameters for the construction of the synaptic matrix
alpha, beta, gamma = SynMatP(a, b, phia, phib)
print(gamma)

# Definition of synaptic matrix
W = SynMat(alpha, beta, gamma)

# Definition of the computation mesh
x = np.linspace(start=0.05, stop=0.95, num=400)
[X, Y] = np.meshgrid(x,x)

# Definition of the vector field mesh
x1 = np.linspace(start=0.05, stop=0.95, num=15)
[X1, Y1] = np.meshgrid(x1,x1)

# Definition of the Energy matrix
E = np.zeros(np.shape(X))

# Definition of the field
U = - X1 + SatTanhF(W[0,0]*X1+W[0,1]*Y1,sl) 
V = - Y1 + SatTanhF(W[1,0]*X1+W[1,1]*Y1,sl) 

#U = - X1 + SigmoidF(W[0,0]*X1+W[0,1]*Y1,sl) 
#V = - Y1 + SigmoidF(W[1,0]*X1+W[1,1]*Y1,sl)

# Computation of the Energy values
for h in range(np.size(x)):
    for k in range(np.size(x)):
        # Definition of useful values

        #inv_phi1, inte1 = Sigmoid(x[h], sl)
        #inv_phi2, inte2 = Sigmoid(x[k], sl)

        inv_phi1, inte1 = SatTanh(x[h], sl)
        inv_phi2, inte2 = SatTanh(x[k], sl)

        x_s = np.array([x[h], x[k]])
        inv_x_s = np.array([inv_phi1, inv_phi2])

        # Computation of point-wise energy value
        E[h,k] = -(1/2)*np.dot(W@x_s,x_s)+np.dot(inv_x_s,x_s)-inte1-inte2

# Plotting values for the colormap
vma = np.max(E)
vmi = np.min(E) 

txt_en = 'EnergyFiring.pdf'

fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(projection='3d')
ax1.set_zlim(vmi-0.4,vma+0.1)
ax1.plot_surface(X,Y,E, cmap=matplotlib.cm.hot, vmin=vmi-0.15, vmax=vma+0.15, lw=0)
ax1.quiver(X1, Y1, vmi-0.4, U, V, np.zeros_like(X1), color='red', alpha=0.5, length=0.05, normalize=True)
ax1.contour(X,Y,E, 75, linewidths=1, cmap="gray", linestyles="solid", offset=vmi-0.4)
ax1.set_xlabel(r'$x_{1}$')
ax1.set_ylabel(r'$x_{2}$')
#ax1.set_zlabel('z')
ax1.set_title(r'$\text{E}_{FR}(x)$')
ax1.grid(False)
#ax1.axis('off')
plt.savefig(txt_en,bbox_inches='tight',format='pdf')
plt.close()

        