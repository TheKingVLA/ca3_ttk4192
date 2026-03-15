from casadi import *
import matplotlib.pyplot as plt
import numpy as np

"""
We want to drive a vehicle from an initial configuration to a final configuration
in minimum time.

States:
    x(t)      : position in x-direction
    y(t)      : position in y-direction
    theta(t)  : vehicle heading

Control inputs:
    v(t)      : forward velocity
    phi(t)    : curvature of the path

Objective:
    Minimize the travel time T:

    min_{v(.), phi(.), T}  T

Subject to the vehicle kinematics:

    x_dot(t)     = v(t) * cos(theta(t))
    y_dot(t)     = v(t) * sin(theta(t))
    theta_dot(t) = v(t) * phi(t)

Boundary conditions:

    x(0)     = changeable
    y(0)     = changeable
    theta(0) = changeable

    x(T)     = changeable
    y(T)     = changeable
    theta(T) = changeable

Control constraints:

    0 <= v(t) <= 1
    -15 <= phi(t) <= 15

for all t in [0, T].

This optimal control problem is solved using CasADi with:
    - Runge-Kutta 4 integration
    - Multiple shooting
    - IPOPT nonlinear solver
"""

# INITIALIZE OPTI STACK
opti = Opti()

# PARAMETERS
N = 100 
T = opti.variable() # The final time T is a variable to be optimized
dt = T/N

# STATES
X = opti.variable(3, N+1) # X = [x; y; theta]
x = X[0,:]
y = X[1,:]
theta = X[2,:]

# CONTROLS
U = opti.variable(2, N) # U = [v; phi]
v = U[0,:]
phi = U[1,:]

# DYNAMICS
def f(x,y,theta,v,phi):
    xdot = v*cos(theta)
    ydot = v*sin(theta)
    thetadot = v*phi
    return vertcat(xdot, ydot, thetadot)

# RUNGE-KUTTA 4 INTEGRATION
for k in range(N):

    xk = X[:,k]
    vk = v[k]
    phik = phi[k]

    k1 = f(xk[0],xk[1],xk[2],vk,phik)
    k2 = f(xk[0]+dt/2*k1[0], xk[1]+dt/2*k1[1], xk[2]+dt/2*k1[2], vk, phik)
    k3 = f(xk[0]+dt/2*k2[0], xk[1]+dt/2*k2[1], xk[2]+dt/2*k2[2], vk, phik)
    k4 = f(xk[0]+dt*k3[0], xk[1]+dt*k3[1], xk[2]+dt*k3[2], vk, phik)

    x_next = xk + dt/6*(k1 + 2*k2 + 2*k3 + k4)

    opti.subject_to(X[:,k+1] == x_next)

# INITIAL CONDITIONS
opti.subject_to(x[0] == 1)
opti.subject_to(y[0] == 1)
opti.subject_to(theta[0] == 2*pi)

# FINAL CONDITIONS
opti.subject_to(x[N] == 0.25)
opti.subject_to(y[N] == 2)
opti.subject_to(theta[N] == pi/4)

# CONTROL CONSTRAINTS (BOUNDED)
opti.subject_to(opti.bounded(0, v, 1))
opti.subject_to(opti.bounded(-15, phi, 15))

# TIME POSITIVE
opti.subject_to(T >= 0)

# OBJECTIVE
opti.minimize(T)

# INITIAL GUESS
opti.set_initial(T,1)
opti.set_initial(x, np.linspace(0,0.25,N+1))
opti.set_initial(y, np.linspace(0,0.25,N+1))
opti.set_initial(theta, np.linspace(0,pi,N+1))

# SOLVER
opti.solver("ipopt") # Using IPOPT

# SOLVE THE PROBLEM
sol = opti.solve()

# SOLUTIONS
x_sol = sol.value(x)
y_sol = sol.value(y)
theta_sol = sol.value(theta)
v_sol = sol.value(v)
phi_sol = sol.value(phi)
T_sol = sol.value(T)

print("Optimal time:",T_sol)

# PLOTTING FROM HERE ONWARD
fig, axs = plt.subplots(3, 1, figsize=(6,10))
t = np.linspace(0,T_sol,N+1)

# TRAJECTORY
axs[0].plot(x_sol, y_sol)
axs[0].set_xlabel("x")
axs[0].set_ylabel("y")
axs[0].set_title("Optimal trajectory (x-y)")
axs[0].axis("equal")
axs[0].grid()

# CURVATURE
axs[1].plot(t[:-1], phi_sol)
axs[1].set_xlabel("time")
axs[1].set_ylabel("curvature φ")
axs[1].set_title("Curvature input")
axs[1].grid()

# VELOCITY
axs[2].plot(t[:-1], v_sol)
axs[2].set_xlabel("time")
axs[2].set_ylabel("velocity v")
axs[2].set_title("Velocity input")
axs[2].grid()

plt.tight_layout()
plt.show()
