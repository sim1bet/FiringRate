# Script for the generation of the plots of activity for both the tensions/currents and
# the brownian motion associated to the timescales

import matplotlib.pyplot as plt
import numpy as np

params = {'ytick.labelsize': 35,
          'xtick.labelsize': 35,
          'axes.labelsize' : 30,
          'font.size' : 20,
          'axes.titlesize': 35}
plt.rcParams.update(params)

# Function for the generation of the overlap graph
def PlotOverlap(y, T, title):

    # Definition of the time axis
    x = np.linspace(start=0, stop=T, num=np.size(y[0,:]))

    # Plotting of the trajectory overlap
    
    fig = plt.figure(figsize=(10,10))
    ax1 = fig.add_subplot()
    for p in range(2):
        if p == 0:
            lbl = r'$\bar \xi^{\nu}$'
        elif p==2 :
            lbl = r'$\bar \xi$'
        else:
            lbl = r'$\bar \xi^{\mu}\neq\bar \xi^{\nu}$'
        ax1.plot(x[1:],y[p,1:], label = lbl, linewidth=8.0)
    ax1.hlines(0.2, xmin=0, xmax=T, color = 'purple', label = 'p',linestyles='dashed',  linewidth = 4.0)
    #ax1.hlines(0.4, xmin=0, xmax=T, color = 'cyan', label = 'r',linestyles='dashed',  linewidth = 4.0)
    ax1.set_ylim(bottom=0,top=1.05) 
    ax1.set_xlabel(r'$t$')
    ax1.set_ylabel(r'memory overlap')
    ax1.legend()
    ax1.set_title(r'Retrieval of memory $\xi^{\nu}$')
    plt.savefig(title, bbox_inches = 'tight', format='pdf')
    plt.close()
