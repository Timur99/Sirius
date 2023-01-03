import random
import numpy as np
import math
import matplotlib.pyplot as plt

N = 100
T = 1
X =[i /N for i in range (N+1)]

Delta = T / N
#сетку

def winir () :
    W = []
    W += [0]
    #сетка для N  xs = np.linspace(a, b, 100)
    N_n = np.random.normal (0, 1, len(X))
    for i in range (1,len( X ) ) :
        W +=[ W [i -1]+ N_n[i] * math.sqrt ( Delta ) ]
    return W


W = winir()
plt.figure ( figsize =(10, 7) )
plt.grid ()
plt.plot (X , W , linewidth =2.0)
plt.title ("Wiener process")
plt.xlabel ("t")
plt.ylabel ("W")
plt.show ()

#random sead


#S = ///
#for 10 100 1000