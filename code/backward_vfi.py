# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Backward induction from terminal value function

# %%
import numpy as np
import scipy.optimize as opt
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import SmoothBivariateSpline
from numba import jit
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import time
import copy

start = time.time()


# %%
# Define g(x) and g'(x) (labor market equilibrium tradeoff) function
@jit
def g(x):
    k = 5/4
    s = 1/2
    a = 1/(k**0.5 - s)
    y = k - (x/a + s)**2
    
    return y

@jit
def g_prime(x):
    k = 5/4
    s = 1/2
    a = 1/(k**0.5 - s)
    y = (-2*(x/a + s))/a
    
    return y

# %% [markdown]
# # Terminal period

# %%
# Define terminal utility
@jit
def utility_T(values, state, params):
    beta, r, a, delta, alpha, gamma, sigma = params # Unpack parameters
    A0, K0 = state # Unpack chosen state
    c, l = values # Unpack values to evaluate at
    
    # Put constraints into utility function
    A1 = (1+r)*A0 + (1-l)*K0 - c
    
    # Evaluate utility
    u = - np.log(c) - alpha*np.log(l) - gamma*np.log(A1)
    return u


# %%
# Define parameters
beta = 0.9 # Discount rate
r = 0.1 # Return on A
a = 1 # Productivity in production of K
delta = 0.1 # Depreciation rate of K
alpha = 1 # Weight on leisure
gamma = 1 # Weight on bequest
sigma = 0.5 # Unused
params = (beta, r, a, delta, alpha, gamma, sigma)


# %%
# Grid parameters; time dependent
T = 5
smoothness = 10
A_min = np.zeros(T)
A_max = np.zeros(T)
A_size = 100 # Number of grid points in every time period
K_min = np.ones(T)
K_max = np.zeros(T)
K_size = 100 # Number of grid points in every time period

A_grid = np.zeros((A_size, T))
K_grid = np.zeros((K_size, T))

# Set up state grids
for t in range(T):
    A_max[t] = 1*(2**t)
    K_max[t] = 2*(2**t)
    
    K_min[t] = (1-delta)**t
    
    A_grid[:, t] = np.linspace(A_min[t], A_max[t], A_size)
    K_grid[:, t] = np.linspace(K_min[t], K_max[t], K_size)

T_grid = np.linspace(0, T, T+1, dtype = int)

# Value grid
V_grid = np.ones((A_size, K_size, T))

# Policy grid
c_policy = np.ones((A_size, K_size, T))
l_policy = np.ones((A_size, K_size, T))
x_policy = np.zeros((A_size, K_size, T))

# Store interpolated value function in dict
V_dict = {}


# %%
# Compute value function and policy at t = T
for a_grid_ind, A in np.ndenumerate(A_grid[:, T-1]):
    for k_grid_ind, K in np.ndenumerate(K_grid[:, T-1]):
        
        a_ind = a_grid_ind[0]
        k_ind = k_grid_ind[0]
        
        # Compute value function at terminal state
        state = (A, K)
        results = opt.minimize(utility_T, [0.1, 0.1], bounds = ((0, None), (0, 1)), args = (state, params,), method = 'SLSQP')
        
        # Store policy results and value function
        V_grid[a_ind, k_ind, T-1] = -results.fun
        c_policy[a_ind, k_ind, T-1] = results.x[0]
        l_policy[a_ind, k_ind, T-1] = results.x[1]


# %%
# Interpolate terminal value function and store in dict
V_dict[T-1] = RectBivariateSpline(A_grid[:, T-1], K_grid[:, T-1], V_grid[:, :, T-1], s = smoothness)


# %%
K, A = np.meshgrid(K_grid[:, T-1], A_grid[:, T-1])


# %%
plt.contourf(A, K, V_grid[:, :, -1], 20, cmap='viridis')
plt.colorbar();


# %%
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(A, K, V_grid[:, :, -1], 50, cmap='binary')
ax.set_title('V')
ax.set_xlabel('A')
ax.set_ylabel('K')
ax.set_zlabel('V');


# %%
ax = plt.axes(projection='3d')
ax.plot_surface(A, K, V_grid[:, :, -1], rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_title('V')
ax.set_xlabel('A')
ax.set_ylabel('K')
ax.set_zlabel('V')


# %%
plt.contourf(A, K, c_policy[:, :, -1], 20, cmap='viridis')
plt.colorbar();


# %%
ax = plt.axes(projection='3d')
ax.plot_surface(A, K, c_policy[:, :, -1], rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_title('c')
ax.set_xlabel('A')
ax.set_ylabel('K')
ax.set_zlabel('V')


# %%
plt.contourf(A, K, l_policy[:, :, -1], 20, cmap='viridis')
plt.colorbar();


# %%
c_policy[:, 0, -1]


# %%
# Inspect bequest
V_test = np.zeros((A_size, K_size))
bequest = np.zeros((A_size, K_size))

for a_grid_ind, A in np.ndenumerate(A_grid[:, T-1]):
    for k_grid_ind, K in np.ndenumerate(K_grid[:, T-1]):
        
        a_ind = a_grid_ind[0]
        k_ind = k_grid_ind[0]
        
        state = (A, K)
        c = c_policy[a_ind, k_ind, T-1]
        l = l_policy[a_ind, k_ind, T-1]
        values = (c, l)
        
        V_test[a_ind, k_ind] = -utility_T(values, state, params)
        bequest[a_ind, k_ind] = (1+r)*A + (1-l)*K - c


# %%
K, A = np.meshgrid(K_grid[:, T-1], A_grid[:, T-1])
ax = plt.axes(projection='3d')
ax.plot_surface(A, K, bequest[:, :], rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_title('Bequest')
ax.set_xlabel('A')
ax.set_ylabel('K')
ax.set_zlabel('V')


# %%
plt.contourf(A, K, bequest[:, :], 20, cmap='viridis')
plt.colorbar();

# %% [markdown]
# # Iterate back

# %%
# Define objective function for t < T
def objective(values, state, V_T_interpolate, params):
    beta, r, a, delta, alpha, gamma, sigma = params # Unpack parameters
    A0, K0 = state # Unpack current state
    c, l, x = values # Unpack values to evaluate at
    
    # State evolution
    A1 = (1+r)*A0 + g(x)*(1-l)*K0 - c
    K1 = (1 + a*x*(1-l) - delta)*K0
    
    # Value next period
    V1 = RectBivariateSpline.__call__(V_T_interpolate, A1, K1, s = smoothness)[0][0] # Continuation value of A1, K1
    
    # Evaluate objective
    u = - np.log(c) - alpha*np.log(l) - beta*V1
    return u


# %%
# Define borrowing constraint function
def no_borrowing_constraint(values, state, params):
    beta, r, a, delta, alpha, gamma, sigma = params # Unpack parameters
    A0, K0 = state # Unpack current state
    c, l, x = values # Unpack values to evaluate at
    
    # Asset next period has to be weakly positive
    A1 = (1+r)*A0 + g(x)*(1-l)*K0 - c
    
    return A1


# %%
# Compute value function and policy at t
for t in range(T-2, -1, -1):
    print(t)
    for a_grid_ind, A in np.ndenumerate(A_grid[:, t]):
        for k_grid_ind, K in np.ndenumerate(K_grid[:, t]):
            
            a_ind = a_grid_ind[0]
            k_ind = k_grid_ind[0]

            state = (A, K)
            
            # Define constraints, including no borrowing constraints
            constraint_list = [{'type':'ineq',
                    'fun':no_borrowing_constraint,
                    'args':(state, params,)},
                   {'type':'ineq',
                    'fun': lambda x: x[0]},
                   {'type':'ineq',
                    'fun': lambda x: x[1]},
                   {'type':'ineq',
                    'fun': lambda x: 1 - x[1]},
                   {'type':'ineq',
                    'fun': lambda x: x[2]},
                   {'type':'ineq',
                    'fun': lambda x: 1 - x[2]}]

            # Compute value function at t
            results = opt.minimize(objective, [0.1, 0.1, 0.1], constraints = constraint_list, args = (state, V_dict[t+1], params,), method = 'SLSQP')

            # Store results
            V_grid[a_ind, k_ind, t] = -results.fun
            c_policy[a_ind, k_ind, t] = results.x[0]
            l_policy[a_ind, k_ind, t] = results.x[1]
            x_policy[a_ind, k_ind, t] = results.x[2]
            
            # Interpolate value function at t and store in dict
            V_dict[t] = RectBivariateSpline(A_grid[:, t], K_grid[:, t], V_grid[:, :, t], s = smoothness)


# %%
end = time.time()
print(end - start)


# %%
# Check results
t = T-4
K, A = np.meshgrid(K_grid[:, t], A_grid[:, t])

ax = plt.axes(projection='3d')
ax.plot_surface(A, K, c_policy[:, :, t], rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_title('c')
ax.set_xlabel('A')
ax.set_ylabel('K')
ax.set_zlabel('c')


# %%
plt.contourf(A, K, c_policy[:, :, t], 20, cmap='viridis')
plt.colorbar();


# %%
ax = plt.axes(projection='3d')
ax.plot_surface(A, K, l_policy[:, :, t], rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_title('l')
ax.set_xlabel('A')
ax.set_ylabel('K')
ax.set_zlabel('c')


# %%
plt.contourf(A, K, l_policy[:, :, t], 20, cmap='viridis')
plt.colorbar();


# %%
ax = plt.axes(projection='3d')
ax.plot_surface(A, K, x_policy[:, :, t], rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_title('x')
ax.set_xlabel('A')
ax.set_ylabel('K')
ax.set_zlabel('c')


# %%
plt.contourf(A, K, x_policy[:, :, t], 20, cmap='viridis')
plt.colorbar();


# %%
plt.scatter(K_grid[:, t], x_policy[70, :, t])


# %%
ax = plt.axes(projection='3d')
ax.plot_surface(A, K, V_grid[:, :, t], rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_title('V')
ax.set_xlabel('A')
ax.set_ylabel('K')
ax.set_zlabel('V')


# %%
plt.contourf(A, K, V_grid[:, :, t], 20, cmap='viridis')
plt.colorbar();

# %% [markdown]
# # Simulate path of variables

# %%
c_path = np.zeros(T)
l_path = np.zeros(T)
x_path = np.zeros(T)
A_path = np.zeros(T)
A_path_actual = np.zeros(T)
K_path = np.zeros(T)
K_path_actual = np.zeros(T)
t = 0


# %%
# Initial state value
A_init = 0.5
K_init = 1


# %%
for t in range(T):
    # Index of current state grid that is closest to given value of A0 and K9
    if t == 0:
        A0_ind = np.abs(A_grid[:, t] - A_init).argmin()
        K0_ind = np.abs(K_grid[:, t] - K_init).argmin()
        A_path[t] = A_grid[A0_ind, t]
        K_path[t] = K_grid[K0_ind, t]
    
    elif t > 0:
        A0_ind = np.abs(A_grid[:, t] - A_path[t]).argmin()
        K0_ind = np.abs(K_grid[:, t] - K_path[t]).argmin()
        
    A0 = A_grid[A0_ind, t]
    K0 = K_grid[K0_ind, t]
    
    # Policies at these grid points closest to given state
    c = c_policy[A0_ind, K0_ind, t]
    l = l_policy[A0_ind, K0_ind, t]
    x = x_policy[A0_ind, K0_ind, t]
    
    # State next period given optimal policy
    A1 = (1+r)*A0 + g(x)*(1-l)*K0 - c
    K1 = (1 + a*x*(1-l) - delta)*K0
    
    # Find point on state grid next period closest to actual state next period, and use that grid point as state next period instead
    if t < T-1:
        A1_ind = np.abs(A_grid[:, t+1] - A1).argmin()
        A_path[t+1] = A_grid[A1_ind, t+1]
        K1_ind = np.abs(K_grid[:, t+1] - K1).argmin()
        K_path[t+1] = K_grid[K1_ind, t+1]
    
    c_path[t] = c
    l_path[t] = l
    x_path[t] = x


# %%
plt.plot(K_path)


# %%


