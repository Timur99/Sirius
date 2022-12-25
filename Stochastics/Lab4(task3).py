from pymle_master.pymle.models import CKLS, OrnsteinUhlenbeck
from pymle_master.pymle.TransitionDensity import EulerDensity, OzakiDensity    #ShojiOzakiDensity
from pymle_master.pymle.fit.AnalyticalMLE import AnalyticalMLE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import matplotlib.ticker as ticker
from scipy.stats import norm
import scipy.stats as st



def ou(X, kappa, alpha, sigma, dW, dt):
    dX = kappa * (alpha - X) * dt + sigma * dW
    return dX


kappa = 3
alpha = 2
sigma = 0.5

N = 1000
delta = 1 / N


param_bounds = [(0, 8), (0, 4), (0.01, 1)]
guess = np.array([2, 1, 0.4])

model = OrnsteinUhlenbeck()

M = 50
param = []

theta1 =[]
theta2 = []
theta3 = []
for j in range(M):
    dW = np.random.normal(0, np.sqrt(1 / N), N)
    X_ou = np.zeros(N)
    X_ou[0] = 5
    for i in range(1, N):
        X_ou[i] = X_ou[i - 1] + ou(X_ou[i - 1], kappa, alpha, sigma, dW[i], delta)
    param = AnalyticalMLE(X_ou, param_bounds, delta, density=OzakiDensity(model)).estimate_params(guess)
    #print(param)
    theta1.append(param.params[0])
    theta2.append(param.params[1])
    theta3.append(param.params[2])

#print(param)
print(st.t.interval(alpha=0.95, df=len(theta1)-1, loc=np.mean(theta1), scale=st.sem(theta1)) )
print(st.t.interval(alpha=0.95, df=len(theta2)-1, loc=np.mean(theta2), scale=st.sem(theta2)) )
print(st.t.interval(alpha=0.95, df=len(theta3)-1, loc=np.mean(theta3), scale=st.sem(theta3)) )
