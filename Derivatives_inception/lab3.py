import lab2
import numpy as np

import matplotlib.pyplot as plt
import math
import random
from math import sqrt, sin, pi



def phi (i ,t , T ) :
    return (2* sqrt (2* T ) ) /(( 2* i +1) * pi ) * sin ((( 2* i +1) * pi * t ) /(2* T ) )

def sum_W (t , T ) :
    Sum =0
    for i in range (len( Z ) ) :
        Sum += Z [ i ]* phi (i ,t , T )
    return Sum

plt . figure ( figsize =(10, 7) )
plt . grid ()

T =1
N =100
t = [ i /N for i in range (N+1) ]
nn = [10,50,100]
n =10
Z = np . random . normal (0, 1, n )

def WWW():
    W =[]
    for i in range ( N +1) :
        W +=[ sum_W ( t [ i ] , T ) ]
    return W

for j in nn():
    plt . plot (t , WWW() , linewidth =2.0)
'''
n =50
Z = np . random . normal (0, 1, n )
W =[]

for i in range ( N +1) :
    W +=[ sum_W ( t [ i ] , T ) ]

plt . plot (t , W , linewidth =2.0)
n =100
Z = np . random . normal (0, 1, n )
W =[]

for i in range ( N +1) :
    W +=[ sum_W ( t [ i ] , T ) ]
'''
#plt . plot (t , W , linewidth =2.0)
#plt . title (’Wiener ␣ process ’)
plt . xlabel ("t")
plt . ylabel ("W")
#plt . legend ([ ’n=10’, ’n=50’, ’n=100’] , loc =’upper ␣ right
#,→ ’)
plt.show()