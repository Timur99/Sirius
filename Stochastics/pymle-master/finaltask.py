




import matplotlib.pyplot as plt
import csv
import pandas as pd
import matplotlib.pyplot as plt


import pandas as pd
import matplotlib.pyplot as plt
from datetime import *
import numpy as np
#from openpyxl import *
import scipy.stats as stats
from pymle.fit.AnalyticalMLE import AnalyticalMLE
from pymle.models import CEV, CIR, OrnsteinUhlenbeck, CKLS, BrownianMotion
from pymle.sim.Simulator1D import Simulator1D
from pymle.TransitionDensity import ExactDensity, EulerDensity, OzakiDensity, ShojiOzakiDensity, KesslerDensity



data0 = pd.read_excel('data.xls')
data = data0.dropna(axis=0, how='any')
x = np.array(data['DEXUSEU'])
N = len(x)
T = N
t = np.linspace(0, T, N)


param = []
delta = 1 / N
dW = np.random.normal(0, np.sqrt(1 / N), N)

def param_of_model(x):
    theta_for_kessler =[]
    theta_for_ozaki = []
    theta_for_euler = []
    aic_for_kessler =[]
    aic_for_ozaki = []
    aic_for_euler = []
    param_bounds = [(0, 10), (0, 5), (0.01, 1)]
    guess = np.array([1, 1, 0.04])
    model = OrnsteinUhlenbeck()
    final_params_for_all = []
    final_aic = []

    kessler = AnalyticalMLE(x, param_bounds, delta, density=KesslerDensity(model)).estimate_params(guess)
    ozaki = AnalyticalMLE(x, param_bounds, delta, density=ShojiOzakiDensity(model)).estimate_params(guess)
    euler = AnalyticalMLE(x, param_bounds, delta, density=EulerDensity(model)).estimate_params(guess)

    final_params_for_all.append(kessler.params)
    final_params_for_all.append(ozaki.params)
    final_params_for_all.append(euler.params)

    final_aic.append(kessler.aic)
    final_aic.append(ozaki.aic)
    final_aic.append(euler.aic)

    return(final_params_for_all, final_aic)



new_params, new_aic = param_of_model(x)




def ou(params_for_ou, delta, dW):

    X_ou = np.zeros(N)
    X_ou[0] = 1.2
    for i in range(1, N):
        X_ou[i] = X_ou[i - 1] + params_for_ou[0] * (params_for_ou[1] - X_ou[i - 1]) * delta + params_for_ou[2] * dW[i]
    return(X_ou)


fig, ax = plt.subplots()
ax.plot(t, x, label='origin_data')
ax.plot(t, ou(new_params[0], delta, dW), label='kessler')
ax.plot(t, ou(new_params[1], delta, dW), label='ozak')
ax.plot(t, ou(new_params[2], delta, dW), label='euler')
plt.grid()
plt.legend()
plt.show()


#for first model OU
print(new_params)
#print('MLE for kessler: ' + new_params[0] + 'AIC: ' + new_aic[0])
print(f'\nMLE for ShojiOzaki: {new_params[1]} \n')
print(f'\nMLE for euler: {new_params[2]} \n')
print(f'\nAIC for kessler: {new_aic[0]}')
print(f'\nAIC for ShojiOzaki: {new_aic[1]}')
print(f'\nAIC for euler: {new_aic[2]}')




if min(new_aic) == new_aic[0]:
    a = 'Model Kessler'
elif min(new_aic) == new_aic[1]:
    a = 'Model ShojiOzaki'
elif min(new_aic) == new_aic[2]:
    a = 'Model Euler'

print("The best model: " + a)

#for second model CKLS



def param_of_model2(x):
    theta_for_kessler2 =[]
    theta_for_ozaki2 = []
    theta_for_euler2 = []
    aic_for_kessler2 =[]
    aic_for_ozaki2 = []
    aic_for_euler2 = []
    param_bounds = [(0.0, 10), (0.0, 10), (0.01, 3), (0.1, 2)]
    guess = np.array([0.01, 0.1, 0.2, 0.6])

    #param_bounds = [(-1, 3), (-1, 5), (0.01, 1), (0.01, 2)]
    #guess = np.array([0.01, 0.01, 0.01, 0.2])
    model2 = CKLS()
    final_params_for_all2 = []
    final_aic2 = []

    kessler2 = AnalyticalMLE(x, param_bounds, delta, density=KesslerDensity(model2)).estimate_params(guess)
    ozaki2 = AnalyticalMLE(x, param_bounds, delta, density=ShojiOzakiDensity(model2)).estimate_params(guess)
    euler2 = AnalyticalMLE(x, param_bounds, delta, density=EulerDensity(model2)).estimate_params(guess)

    final_params_for_all2.append(kessler2.params)
    final_params_for_all2.append(ozaki2.params)
    final_params_for_all2.append(euler2.params)

    final_aic2.append(kessler2.aic)
    final_aic2.append(ozaki2.aic)
    final_aic2.append(euler2.aic)

    return(final_params_for_all2, final_aic2)



new_params2, new_aic2 = param_of_model2(x)


def ckls(params_for_ckls, delta, dW):
    #dW = np.random.normal(0, np.sqrt(1 / N), N)
    X_ckls = np.zeros(N)
    X_ckls[0] = 1.2
    for i in range(1, N):
        X_ckls[i] = X_ckls[i - 1] + (params_for_ckls[0] + params_for_ckls[1] * X_ckls[i - 1]) * delta + params_for_ckls[2] * X_ckls[i - 1] ** params_for_ckls[0] * dW[i]
    return(X_ckls)


fig, ax = plt.subplots()
ax.plot(t, x, label='origin_data')
ax.plot(t, ckls(new_params2[0], delta, dW), label='kessler')
ax.plot(t, ckls(new_params2[1], delta, dW), label='ozak')
ax.plot(t, ckls(new_params2[2], delta, dW), label='euler')
plt.grid()
plt.legend()
plt.show()


#for first model OU
print(new_params2)
print(f'\nMLE for kessler: {new_params2[0]} \n')
print(f'\nMLE for ShojiOzaki: {new_params2[1]} \n')
print(f'\nMLE for euler: {new_params2[2]} \n')
print(f'\nAIC for kessler: {new_aic2[0]}')
print(f'\nAIC for ShojiOzaki: {new_aic2[1]}')
print(f'\nAIC for euler: {new_aic2[2]}')


if min(new_aic2) == new_aic2[0]:
    a = 'Model Kessler'
elif min(new_aic2) == new_aic2[1]:
    a = 'Model ShojiOzaki'
elif min(new_aic2) == new_aic2[2]:
    a = 'Model Euler'

print("The best model: " + a)


#CIR
#dx = params[0] * (params[1] - x[i-1]) * T / N + params[2] * np.sqrt(x[i-1]) * dw
#x[i] = x[i-1] + dx


print(new_params2)
df = pd.DataFrame([['OU', 'параметр1', 'параметр2', 'параметр3', 'параметр4', 'AIC'],
                    ['Kessler', new_params[0][0], new_params[0][1], new_params[0][2], '-', new_aic[0]],
                   ['ShojiOzaki', new_params[1][0], new_params[1][1], new_params[1][2], '-', new_aic[1]],
                   ['Euler', new_params[2][0], new_params[2][1], new_params[2][2], '-', new_aic[2]],
                   ['CKLS', '-','-','-','-','-'],
                    ['Kessler', new_params2[0][0], new_params2[0][1], new_params2[0][2], new_params2[0][3], new_aic2[0]],
                    ['ShojiOzaki', new_params2[1][0], new_params2[1][1], new_params2[1][2], new_params2[1][3], new_aic2[1]],
                    ['Euler', new_params2[2][0], new_params2[2][1], new_params2[2][2], new_params2[2][3], new_aic2[2]]
                   ])

print(df)