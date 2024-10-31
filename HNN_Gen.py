# Scripts that implements the Hopfield Network Class
# from which to create instances of an Hopfield Network either with
# Orthogonal binary memory patterns
# Random binary memory patterns
import numpy as np
import random

class HNN:
    # Initializer for the Hopfield Class
    def __init__(self, p, eps):
        # Definition of the population size as a power of 2
        self.N = 1024
        # Definition of the probability of activation
        self.p = p
        # Definition of the admissible number of memories
        self.P = np.int(np.floor(self.N/(10*np.log(self.N))))
        # Definition of the magnitude of the perturbation of the initial condition
        self.eps = eps

    # Function for the generation of the population and the associated memories
    def net(self, D):
        
        # Generation of the memories either as dependent or independent vectors
        # D = 'I' --> Independent
        # D = 'NI' --> dependent
        if D == 'I':
            C = np.random.choice([0,1], (self.N,self.N), p=[1-self.p, self.p])
            # Extraction of P patterns
            # Selection of P random non-repeated indices
            idx_b = np.random.permutation(range(self.N-1))
            idx = idx_b[:self.P]
            idx_c = idx_b[self.P:self.P+5]

            # Generation of the memories
            self.mems = C[:,idx]
            self.IC = C[:,idx_c]
        elif D == 'NI':
            # Fix the number of memories
            self.P = 7
            # Generate the memory matrix and the  
            self.mems = np.zeros((self.N, self.P))
            # Construction of the first memory
            idx_act = random.sample(range(2056), 408)
            self.mems[idx_act,0] = 1
            
            # Iterative construction of the remaining random dependent memories
            for m in range(1,self.P):
                v = np.zeros((self.N,))
                # Construction of the compounded memory
                for s in range(m):
                    v += self.mems[:,s]
                # Extraction of the indices for correlation
                sig = np.abs(0*(m-1)*np.random.randn())
                idx_cor_pot = np.nonzero(v>m-1-sig)[0]
                idx_ncor_pot = np.nonzero(v==0)[0]
                # Extraction of the location of the correlated indices
                np.random.shuffle(idx_cor_pot)
                idx_cor = idx_cor_pot[:164]#164
                # Extraction of the location of the non-correlated indices
                np.random.shuffle(idx_ncor_pot)
                idx_ncor = idx_ncor_pot[:244]#244
                
                # Setting of the new memory
                self.mems[idx_cor,m] = 1
                self.mems[idx_ncor,m] = 1
                
            self.IC = np.random.choice([0,1], (self.N,3), p=[1-self.p, self.p])#self.mems[:,-2:]#
            self.mems = self.mems[:,:-2]
            self.P = self.P-2
                
            

    # Function that generates the initial condition for the dynamics
    def y0_gen(self):
        
        # generation of the initial condition
        self.y0 = np.multiply(self.mems[:,1],np.random.choice([0,1], size=(self.N,), p=[0.5, 0.5]))
