import numpy as np

import matplotlib.pyplot as plt
from math import sqrt, exp
import random


r =1
sigma =0.5
x =10
N =100
T =1
Delta = T / N
W =[0]
t = list ( np . linspace (0,T , N +1) ) #сетка времени

for i in range ( N ) : #винеровский процесс
    W +=[ W [i -1]+ np . random . normal (0, 1, 1) * sqrt ( Delta ) ]

S = []
for i in range (len(W)):
    S.append(x * exp (( r - sigma **2/2) * t [ i ] + sigma * W [ i ])) #for i in range (len( W ) ) ] #броуновское движение

plt . figure ( figsize =(10, 7) )
plt . grid ()
plt . plot (t , S , linewidth =2.0)

plt . title ("geometric ␣ Brownian ␣ motion ")
plt . xlabel ("t")
plt . ylabel ("S")
plt . show ()