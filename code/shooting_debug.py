# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Try shooting method again

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# %% [markdown]
# Labor market tradeoff function

# %%
# Define g(x) and g'(x) (labor market equilibrium tradeoff) function

def g(x):
    y = 1.25 - (x*(1.25**0.5 - 0.5) + 0.5)**2
    
    if y <= 0:
        return 0
    
    return y

def g_prime(x):
    y = - 2*(x*(1.25**0.5 - 0.5) + 0.5)*(1.25**0.5 - 0.5)
    
    return y


# %%
# Evaluate derivative at corners
g_prime_0 = g_prime(0)
g_prime_1 = g_prime(1)

# %% [markdown]
# Set parameters

# %%
# Parameters
rho = 0.03
r = 0.05
a = 1
delta = 0.05
c_param = 0.5
l_param = 0.5
B_param = 0.5
params = (rho, r, a, delta, c_param, l_param, B_param)

T = 70

# %% [markdown]
# Utility functions and derivatives

# %%
# Define utility functions
def U(c, c_param):
    u = (c**c_param)/c_param
    return u

def U_prime(c, c_param):
    u_prime = c**(c_param - 1)
    return u_prime

def L(l, l_param):
    u = (l**l_param)/l_param
    return u

def L_prime(l, l_param):
    u_prime = l**(l_param - 1)
    return u_prime

def B(A, B_param):
    u = (A**B_param)/B_param
    return u

def B_prime(A, B_param):
    u_prime = A**(B_param - 1)
    return u_prime

# %% [markdown]
# Define a function that produces K1, A1, mu1, p1, c, l, and x, given initial values K0, A0, mu0, p0

# %%
def simulate(K0, A0, mu0, p0):
    # rho and r pins down evolution of mu
    mu1 = mu0*(rho - r) + mu0

    # Value of mu pins down value of c
    c = mu0**(1/(c_param-1))

    # Check for retirement
    x_test = np.linspace(0, 1, 100)
    mc_leisure = np.zeros(100)
    for i, x in enumerate(x_test):
        mc_leisure[i] = mu0*K0*g(x) + a*p0*K0*x

    if np.amax(mc_leisure) > 1:
        retirement = False
    else:
        retirement = True

    # If non-retired, then need to compute value of x and l
    # Check corners
    # x = 0 corner (no investment in human capital)
    if a*p0*K0 + mu0*K0*g_prime_0 < 0:
        no_learning = True
    else:
        no_learning = False

    # x = 1 corner (in school)
    if a*p0*K0 + mu0*K0*g_prime_1 > 0:
        in_school = True
    else:
        in_school = False

    # Evaluate at interior if not at corner; otherwise use corner values
    if no_learning == False and in_school == False:
        interior_x = True
    else:
        interior_x = False

    # Compute value of l if x is at corner
    if interior_x == False:
        if in_school == True: # in school means x = 1
            l = (a*p0*K0)**(1/(l_param-1))
            x = 1
        
        if no_learning == True: # no learning means x = 0
            l = (mu0*K0)**(1/(l_param-1))
            x = 0

    # If l and x are interior, the values of p and mu pins down x, which then determines l
    if interior_x == True:
        x = ((a*p0/mu0)/(2*(1.25**0.5 - 0.5)) - 0.5)/(1.25**0.5 - 0.5)
        l = (mu0*K0*g(x) + a*p0*K0*x)**(1/(l_param-1))

    # Evolution of p
    h = 1 - l
    if retirement == True: # If l = 1 (retired), x and l does not matter
        p1 = p0*(rho + delta) + p0
    elif retirement == False:
        p1 = p0*(rho + delta - a*x*h) - g(x)*h*mu0 + p0

    # Evolution of K and A
    if retirement == True:
        A1 = r*A0 - c + A0
        K1 = -delta*K0 + K0
    elif retirement == False:
        A1 = r*A0 + g(x)*h*K0 - c + A0
        K1 = (a*x*h - delta)*K0 + K0

    return K1, A1, mu1, p1, c, l, x


# %%
K_path = np.zeros(T+1)
A_path = np.zeros(T+1)
mu_path = np.zeros(T+1)
p_path = np.zeros(T+1)
c_path = np.zeros(T)
l_path = np.zeros(T)
x_path = np.zeros(T)


# %%
# Initial conditions
# Initial state
K_path[0] = 1
A_path[0] = 1

# Initial costate
mu_path[0] = 10
p_path[0] = 1000


# %%
# Iterate forward in time
for t in range(T):
    K0 = K_path[t]
    A0 = A_path[t]
    mu0 = mu_path[t]
    p0 = p_path[t]

    K1, A1, mu1, p1, c, l, x = simulate(K0, A0, mu0, p0)

    K_path[t+1] = K1
    A_path[t+1] = A1
    mu_path[t+1] = mu1
    p_path[t+1] = p1
    c_path[t] = c
    l_path[t] = l
    x_path[t] = x


# %%
# Evaluate terminal conditions
# Terminal states
AT = A_path[T-1]
KT = K_path[T-1]
muT = mu_path[T-1]
pT = p_path[T-1]

cT = c_path[T-1]
bequest = AT - cT


# %%
B_prime(bequest, B_param)


# %%
1/(bequest**0.5)


# %%


