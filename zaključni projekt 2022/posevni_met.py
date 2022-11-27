from random import random
from tkinter import Y
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from numpy import linalg as LA
import random
import mpl_toolkits
from mpl_toolkits import mplot3d



# 1.del ----------------------------------------------------------
# Začetni parametri
S = 3 * (.01)**2
m = 70
rho = 1.2
g = 10 
v_0 = 200

# r = [x, y]
# v = [sqrt(x'**2 +y'**2), sqrt(x'**2 +y'**2)]
# a = [- b/m* sqrt(), -g -b/m*sqrt()]

# zamenjava spremenljivk 42nd order _> 4 1st order, coupled
# x' = v_x
# v_x' = -b* sqrt(x'**2 +y'**2)m* x'
# y' = v_y
# v_y' = -b* sqrt(x'**2 +y'**2)m* y'
# s = [x', v_x', y', v_y']

b1 = 0.5 * 0.45 * rho * S
b = b1

def function1(t, s, b):
    x, vx, y, vy = s
    return np.array([vx, - (b/m) * np.sqrt(vx**2 +vy**2) * (vx/LA.norm(vx)), vy, -g - (b/m) * np.sqrt(vx**2 +vy**2) * (vy/LA.norm(vy))])

fig = plt.figure()
ax = fig.gca()

# = [10, 150, 200, 250, 500]
koti = [ 22, 22.5, 23 ]
#koti = 44.5 + np.array([0.1 * i for i in range(10)])
max_dometi = []

for i, kot in enumerate(koti):
    k = kot * np.pi / 180
    y0 = np.array([0, v_0* np.cos(k), 0, v_0* np.sin(k)])
    print(y0)
    t = (0,45)
    hmax = 0.01

    sol = solve_ivp(function1, t, y0, args=(b,), method="RK45", max_step = hmax)
    t1 = sol.t
    states = sol.y
    x1 = states[0]
    y1 = states[2]
    vx1 = states[1]
    vx2 = states[3]
    ax.plot(x1, y1, label =  str(kot) + r"$^{\circ}$")

    # max domet 
    ind = np.where(y1 < 0)
    index = ind[0][0]
    max_dometi.append(x1[index])
"""
   # v .txt datoteko
    data = open("podatki_veter.txt", "w")
    for el in x1:
        data.write(str(el ) + "\n")
    data.write("y")
    for el in y1:
        data.write(str(el ) + "\n")
    data.write("naslednji")

"""


print(max_dometi)
ax.set(title = "Upor v zraku na kroglo", xlabel = "x[m]", ylabel="z[m]")
plt.xlim([0, 5000])
plt.ylim([0, 2000])

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, fontsize = 10)

plt.legend()

# 2.del ------------------------------------------------------------
# horizontalni vektor hitrosti
c_max = 15
b = 0.5 * rho * S
#b = 2

def function2(t, s, b, v):
    x, vx, y, vy = s
    return np.array([(vx-v), - (b/m) * np.sqrt((vx-v)**2 +vy**2) * vx, vy, -g - (b/m) * np.sqrt((vx-v)**2 +vy**2) * vy])

fig2 = plt.figure()
ax2 = fig2.gca()

koti_2 = [25, 45, 65]
barve_2 = ["orange", "red", "lime", "green", "darkviolet", "blue"]
dometi = []

for i, kot in enumerate(koti_2):
    k = kot * np.pi / 180
    y0 = np.array([0, v_0* np.cos(k), 0, v_0* np.sin(k)])
    t = (0,45)
    hmax = 0.01

    v = round(random.uniform(0, c_max), 2)
    v_ = round(random.uniform(0, c_max), 2)
    v__ = round(random.uniform(0, c_max), 2)
    v___ = round(random.uniform(0, c_max), 2)

    sol_n = solve_ivp(function1, t, y0, args=(b,), method="RK45", max_step = hmax)

    sol = solve_ivp(function2, t, y0, args=(b,v,), method="RK45", max_step = hmax)
    sol_ = solve_ivp(function2, t, y0, args=(b,v_,), method="RK45", max_step = hmax)
    sol__ = solve_ivp(function2, t, y0, args=(b,v__,), method="RK45", max_step = hmax)
    sol___ = solve_ivp(function2, t, y0, args=(b,v___,), method="RK45", max_step = hmax)

    tn = sol_n.t
    states_n = sol_n.y
    x1_n = states_n[0]
    y1_n = states_n[2]


    states = sol.y
    x1 = states[0]
    y1 = states[2]

    states_ = sol_.y
    x1_ = states_[0]
    y1_ = states_[2]

    states__ = sol__.y
    x1__ = states__[0]
    y1__ = states__[2]

    states___ = sol___.y
    x1___ = states___[0]
    y1___ = states___[2]

    ax2.plot(x1_n, y1_n, label =  str(kot) + r"$^{\circ}$")

    ax2.plot(x1, y1,c = "dimgray", label =  str(kot) + r"$^{\circ}$" + " veter {}m/s".format(v))
    ax2.plot(x1_, y1_,c = "lightgrey", label =  str(kot) + r"$^{\circ}$" + " veter {}m/s".format(v_))
    ax2.plot(x1__, y1__,c = "grey", label =  str(kot) + r"$^{\circ}$" + " veter {}m/s".format(v__))
    ax2.plot(x1___, y1___,c = "black", label =  str(kot) + r"$^{\circ}$" + " veter {}m/s".format(v___))


    # Domet
    ind = np.where(y1 < 0)
    index = ind[0][0]
    dometi.append(x1[index])


ax2.set(title = "Trajektorije pod vplivom vetra", xlabel = "x[m]", ylabel="z[m]")
plt.xlim([0, 6000])
plt.ylim([0, 3000])

handles, labels = ax2.get_legend_handles_labels()
ax2.legend(handles, labels, fontsize = 8)

plt.legend()

# 3.del ------------------------------------------------------------------------------
# Sedaj upoštevamo še Coriolisa v 3d

fig3 = plt.figure()
ax3 = fig3.add_subplot(projection='3d')

# coriolis 
w_z = 1/(24*60*60)
theta = 46 * np.pi /180

# funkcija
def function3(t, s, b):
    omega = [0,w_z * np.cos(theta),w_z * np.sin(theta)]
    x, vx, y, vy , z, vz = s
    v_rel = [vx, vy, vz]
    cross = np.cross(omega, v_rel)

    i = cross[0]
    j = cross[1]
    k = cross[2]

    return np.array([(vx), - (b/m) * np.sqrt((vx)**2 +(vy)**2 + (vz)**2) * vx - 2*i,(vy), - (b/m) * np.sqrt((vx)**2 +(vy)**2 + (vz)**2) * vy - 2*j, (vz), -g - (b/m) * np.sqrt((vx)**2 +(vy)**2 + (vz)**2) * vz + 2*k])


koti_3 = [15, 30, 45, 60, 75, 90]
#koti_3 = 20 + np.array([i for i in range(10)])
barve_3 = ["violet", "orange", "lime", "green", "darkviolet", "blue"]
dometi_3 = []

for i, kot in enumerate(koti_3):
    k = kot * np.pi / 180
    y0 = [0, v_0* np.cos(kot),0, 0, 0, v_0* np.sin(kot)]
    t = (0,30)
    hmax = 0.01

    sol3 = solve_ivp(function3, t, y0, args=(b,), method="RK45", max_step = hmax)

    # primerjava
    y0_primerjava = np.array([0, v_0* np.cos(k), 0, v_0* np.sin(k)])

    sol_n = solve_ivp(function1, t, y0_primerjava, args=(b,), method="RK45", max_step = hmax)
    tn = sol_n.t
    states_n = sol_n.y
    x1_n = states_n[0]
    y1_n = states_n[2]
    vx1_n = states_n[1]
    vx2_n = states_n[3]
    z1_n = np.zeros(len(x1_n))

    t1 = sol3.t
    states = sol3.y
    x3 = states[0]
    y3 = states[2]
    z3 = states[4]
    vx3 = states[1]
    vx3 = states[3]
    vz3 = states[5]

    ax3.plot(x3, y3, z3,  label =  str(kot) + r"$^{\circ}$")
    ax3.plot(x1_n, z1_n, y1_n, color = "black")

    # Domet
    ind = np.where(y1 < 0)
    index = ind[0][0]
    dometi_3.append(x1[index])

ax3.set(title = "Upor in Coriolis", xlabel = "x[m]", ylabel="y[m]",zlabel="z[m]" )

handles, labels = ax3.get_legend_handles_labels()
ax3.legend(handles, labels, fontsize = 10)

plt.legend()
plt.show()
