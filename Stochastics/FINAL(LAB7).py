
import pandas as pd
import matplotlib.pyplot as plt
from datetime import *
import numpy as np
import scipy.stats as stats
from pymle_master.pymle.fit.AnalyticalMLE import AnalyticalMLE
from pymle_master.pymle.models import CEV, CIR, OrnsteinUhlenbeck, CKLS, BrownianMotion
from pymle_master.pymle.sim.Simulator1D import Simulator1D
from pymle_master.pymle.TransitionDensity import ExactDensity, EulerDensity, OzakiDensity, ShojiOzakiDensity, KesslerDensity



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

# CIR model


def param_of_model3(x):
    theta_for_kessler3 =[]
    theta_for_ozaki3 = []
    theta_for_euler3 = []
    aic_for_kessler3 =[]
    aic_for_ozaki3 = []
    aic_for_euler3 = []
    param_bounds = [(0, 10), (0, 5), (0.01, 1)]
    guess = np.array([1, 1, 0.04])
    model = CIR()
    final_params_for_all3 = []
    final_aic3 = []

    kessler3 = AnalyticalMLE(x, param_bounds, delta, density=KesslerDensity(model)).estimate_params(guess)
    ozaki3 = AnalyticalMLE(x, param_bounds, delta, density=ShojiOzakiDensity(model)).estimate_params(guess)
    euler3 = AnalyticalMLE(x, param_bounds, delta, density=EulerDensity(model)).estimate_params(guess)

    final_params_for_all3.append(kessler3.params)
    final_params_for_all3.append(ozaki3.params)
    final_params_for_all3.append(euler3.params)

    final_aic3.append(kessler3.aic)
    final_aic3.append(ozaki3.aic)
    final_aic3.append(euler3.aic)

    return(final_params_for_all3, final_aic3)



new_params3, new_aic3 = param_of_model3(x)


def CIR(params_for_ckls, delta, dW):
    X_cir = np.zeros(N)
    X_cir[0] = 1.2
    for i in range(1, N):
        X_cir[i] = X_cir[i-1] + params_for_ckls[0] * (params_for_ckls[1] - X_cir[i-1]) * delta + params_for_ckls[2] * np.sqrt(X_cir[i-1]) *dW[i]
    return(X_cir)

fig, ax = plt.subplots()
ax.plot(t, x, label='origin_data')
ax.plot(t, CIR(new_params3[0], delta, dW), label='kessler')
ax.plot(t, CIR(new_params3[1], delta, dW), label='ozak')
ax.plot(t, CIR(new_params3[2], delta, dW), label='euler')
plt.grid()
plt.legend()
plt.show()




print(new_params2)
df = pd.DataFrame([['OU', 'параметр1', 'параметр2', 'параметр3', 'параметр4', 'AIC'],
                    ['Kessler', new_params[0][0], new_params[0][1], new_params[0][2], '-', new_aic[0]],
                   ['ShojiOzaki', new_params[1][0], new_params[1][1], new_params[1][2], '-', new_aic[1]],
                   ['Euler', new_params[2][0], new_params[2][1], new_params[2][2], '-', new_aic[2]],
                   ['CKLS', '-','-','-','-','-'],
                    ['Kessler', new_params2[0][0], new_params2[0][1], new_params2[0][2], new_params2[0][3], new_aic2[0]],
                    ['ShojiOzaki', new_params2[1][0], new_params2[1][1], new_params2[1][2], new_params2[1][3], new_aic2[1]],
                    ['Euler', new_params2[2][0], new_params2[2][1], new_params2[2][2], new_params2[2][3], new_aic2[2]],
                    ['CIR', '-','-','-','-','-'],
                   ['Kessler', new_params3[0][0], new_params3[0][1], new_params3[0][2], '-', new_aic3[0]],
                   ['ShojiOzaki', new_params3[1][0], new_params3[1][1], new_params3[1][2], '-', new_aic3[1]],
                   ['Euler', new_params3[2][0], new_params3[2][1], new_params3[2][2], '-', new_aic3[2]],
                   ])

print(df)

print(new_params, new_params2, new_params3)
print(new_aic, new_aic2, new_aic3)

