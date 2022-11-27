import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Podatki 
n = 1000 
hmax = 1e-3
omega = 50 
R = 200 
L = 0.4
C = 1e-3 

# 1., 2. DEL ------------------------------------
# solve_ivp
def drugi_odvod(t, e, v):
	a = -v/(R*C) - e/(L*C) + omega/R*np.cos(omega*t) 
	return a

def funkcija(t, y):
    a = drugi_odvod(t, y[0], y[1])
    return np.array([y[1], a])

y0 = np.array([0.0, 0.0]) 
t1 = (0, 1)

s = solve_ivp(funkcija, t1, y0, method='RK45', max_step=hmax)

times = s.t
states = s.y
e = states[0]
v = states[1]

t1 = np.linspace(0,1,1002)

# Grafi
# 1 -et
fig = plt.figure()
ax = fig.gca()
ax.plot(t1, e, label="e(t)")
ax.set(xlabel = "t[s]", ylabel="e[As]")
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, fontsize = 10)


plt.savefig("et.pdf")
plt.legend()

# 2 -iu
U = np.sin(omega*times)
I = (U - e/C) / R

fig2 = plt.figure()
ax2 = fig2.gca()
ax2.plot(I, U, 'r-',label="I(U)")
ax2.set(xlabel = "U[V]", ylabel="I[A]")
handles, labels = ax2.get_legend_handles_labels()
ax2.legend(handles, labels, fontsize = 10)


plt.savefig("IU.pdf")
plt.legend()

# 3. DEL ----------------------------
t2 = (0, 8)
s = solve_ivp(funkcija, t2, y0, method='RK45', max_step=hmax)


times = s.t
states = s.y
e = states[0]
v = states[1]

U = np.sin(omega*times)
beta = 1/(2*R*C)
krivulja = 0.005*np.exp(-beta*times)

# Graf 3.del
P = abs((U - e/C) / R * U)

fig3 = plt.figure()
ax3 = fig3.gca()
ax3.plot(times, P, 'g-', label="|P(t)|")
ax3.plot(times, krivulja, 'y-', label="ovojnica")

plt.yscale('log')
plt.xlabel("t[s]")
plt.ylabel("|P(t)|")
plt.ylim(1e-11, 1e-2)
plt.legend()
plt.savefig("Pt.pdf") 
plt.show()
