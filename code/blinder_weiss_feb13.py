"""
Created on Thu Feb 13 00:01:40 2020

@author: Carlos
"""

import numpy as np
from scipy import optimize, misc

# Define parameters
beta = .99
rho = 0.01
r = 0.05
a = 1
delta = 0.05
alpha = 0.5
gamma = 0.5
sigma = 0.5
params = (rho, r, a, delta, alpha, gamma, sigma)

# Initial conditions
K0 = 1
A0 = 100
mu_0 = 1
p_0 = 1
state_0 = (K_0, A_0, mu_0, p_0)
T = 30

# Define function g(x)

def g(x):
    k = 5/4
    s = 1/2
    a = 1/(k**0.5 - s)
    y = k - (x/a + s)**2
    
    if y <= 0:
        return 0
    
    return y


def g_prime(x):
    k = 5/4
    s = 1/2
    a = 1/(k**0.5 - s)
    y = (-2*(x/a + s))/a
    
    return y


#plt.plot(x,[g(i) for i in x])

# Define functions

def u(c, params=params):
    rho, r, a, delta, alpha, gamma, sigma = params # Unpack parameters    
    return np.log(c)


def v(l, params=params):
    rho, r, a, delta, alpha, gamma, sigma = params # Unpack parameters   
    return np.log(l)


# Define derivatives
u_prime = lambda x: 1/x
v_prime = lambda x: 1/x


# Define FOCS:

def focs(guess,params,T,K0,A0):
    rho, r, a, delta, alpha, gamma, sigma = params # Unpack parameters
    
    # unpack guesses
    guess = np.reshape(guess,[3,T]).T
    c = guess[:,0]
    l = guess[:,1]
    x = guess[:,2]
    
    # solve for states
    A = np.zeros([T,1])
    A[0] = A0
    K = np.zeros([T,1])
    K[0] = K0
    
    for i in range(1,T):
        K[i] = (1-delta-a*x[i]*(1-l[i]))*K[i-1]
        A[i] = (1+r)*A[i-1] + g(x[i])*(1-l[i])*K[i]
    T = c.size
    
    equations_c = np.zeros([T,1])
    equations_c[-1] = float(u_prime(c[-1]) - u_prime((1+r)*A[-1]+g(0)*(1-l[-1])*K[-1]))
    for i in range(0,T-1):
        equations_c[i] = u_prime(c[i]) - (1+r)*beta*(u_prime(c[i+1]))

    equations_l = np.zeros([T,1])
    for i in range(0,T):
        equations_l[i] = v_prime(l[i]) - u_prime(c[i])*(g(x[i])*K[i] - g_prime(x[i])*x[i]*K[i])

    equations_x = np.zeros([T,1])
    equations_x[-1] = float(g_prime(x[-1]) - g_prime(0))
    for i in range(0,T-1):
        equations_x[i] = g_prime(x[i]) - (1+r)*g_prime(x[i+1])*(1-delta+a*x[i+1]*(1-l[i+1])-a*g(x[i+1])*(1-l[i+1]))

    output = np.column_stack((equations_c,equations_l,equations_x)).T.reshape([3*T])
    print(output)
    return output

guess = np.outer(np.ones([T,1]),[3,0.5,0.5]).T.reshape([3*T])
result = guess.copy()
result = optimize.fsolve(focs, guess, args = (params,T,K0,A0))
result = np.reshape(result,[3,T]).T
plt.plot(np.linspace(0,T,T),result[:,1])