import numpy as np
import matplotlib.pyplot as plt
import math
import random



def runif ( n ) :                                   #генерим случайные величины и считаем частичную сумму
    #L = []
    L = np.sign(np.random.normal(0, 1, n))
    #for i in l:
     #   if i > 0:
      #      L += 1
       # else:
        #    L += -1
        #L +=[2*( random . uniform (0, 1) >0.5) -1]
        #L += [2 * (random.uniform(0, 1) > 0.5) - 1]
    return L

#sign x - ?

def function ( x,n ) :
    if x *n >0:
        return S [int ( x * n ) -1]
    else :
        return 0

plt . figure ( figsize =(10, 7) )
plt . grid ()

T = 1
t = [ i /100 for i in range (101) ]
nn = [10,100,1000]
#S = np . cumsum ( runif ( n ) )



fig, ax = plt.subplots()

for j in nn:
#def WW(j):
    S = np.cumsum(runif(j))
    W = [function(x,j) / math.sqrt(j) for x in t]
    #ax.plot(t, W, lw=2)
    ax.plot(t, W, linewidth=2.0)
    ax.grid(linewidth=1)
    #return W
plt.show()


'''    
n = 10
T = 1
t = [ i /100 for i in range (101) ]
S = np . cumsum ( runif ( n ) )
W = [ function ( x ) / sqrt ( n ) for x in t ]

plt . plot (t , W , linewidth =2.0)

n = 100
T = 1
t = [ i /100 for i in range (101) ]
S = np . cumsum ( runif ( n ) )
W = [ function ( x ) / sqrt ( n ) for x in t ]
plt . plot (t , W , linewidth =2.0)

n = 1000
T = 1
t = [ i /100 for i in range (101) ]
S = np . cumsum ( runif ( n ) )
W = [ function ( x ) / sqrt ( n ) for x in t ]
'''

#plt . plot (t , WW(nn[0]) , linewidth =2.0)
#plt . plot (t , WW(nn[1]) , linewidth =2.0)
#plt . plot (t , WW(nn[2]) , linewidth =2.0)
#plt . title ('Wiener ␣ process ')
#plt . xlabel ("t")
#plt . ylabel ("W")
#plt.legend([10,100,1000], loc ='upper right')
#plt . show ()



