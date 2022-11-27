import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import sys, threading
sys.setrecursionlimit(10**7) # max depth of recursion
threading.stack_size(2**27)  # new thread will get stack of such size
import collections
import pandas as pd


# Prvi del
n = 256
t = 128

s_z = np.zeros(n)
for i, el in enumerate(s_z):
    if i == n/2:
        s_z[i] = 1

#pravilo
vpisna = 28201136 
#vpisna = 2820113694
"""pravilo = [0 if i == "b" else int(i) for i in bin(vpisna)]
pravilo_1 = pravilo[2::]
"""
def bin_array(num, dolzina):
    #return np.array(list(np.binary_repr(num).zfill(m))).astype(np.int8)
    return np.array(list(np.binary_repr(num, width=dolzina))).astype(np.int8)

pravilo_1 = bin_array(vpisna, 32)
pravilo_1 = pravilo_1[::-1]

mat = []
def prvi_del(s, p, t):
    while t > 0:
        mat.append(s)
        s_nov = np.zeros(n)
        for i in range(n):
            c = 16 * s[(i + n - 2)%n] + 8 * s[(i + n - 1)%n] + 4 * s[i] + 2 * s[(i + 1)%n] + s[(i + 2)%n]
            s_nov[i] = p[int(c)]
        
        s = s_nov
        t -= 1
prvi_del(s_z, pravilo_1, t)

plt.matshow(mat, cmap="Greys")
plt.xlabel("indeks polja - i")
plt.ylabel("koraki - t")
plt.savefig("mat.pdf")

# Tekstovna datoteka
"""# Opomba: v datoteki ki sem jo poslala vsaka vrstica ni prevedena iz arraya v navaden zapis ker ne znam modula pandas instalirat - 
datoteka = open("mat.dat", "w")

for el in mat:
    cn = el.DataFrame(el).T
    datoteka.write(cn + "\n")
"""
# Drugi del 
n_2 = 2048
t_2 = 1024

s_z_2 = np.zeros(n_2)
for i, el in enumerate(s_z_2):
    if i == n/2:
        s_z_2[i] = 1
        

def drugi_del(s, p, t):
    st_enic = []
    st_nicel = []
    st_korakov = [i for i in range(t)]
    while t > 0:
        s_nov = np.zeros(n)
        for i in range(n-1):
            c = 16 * s[(i + n - 2)%n] + 8 * s[(i + n - 1)%n] + 4 * s[i] + 2 * s[(i + 1)%n] + s[(i + 2)%n]
            s_nov[i] = p[int(c)]

        s = s_nov
        prestevki = collections.Counter(s)
        st_nicel.append(prestevki[0])
        st_enic.append(prestevki[1])
        
        t -= 1
    

    k = np.array(s)
    max_st_enic = np.ones(t_2)
    max_st_nicel = np.ones(t_2)
    m_1 = 0
    m_0 = 0
    z_1 = 0
    z_0 = 0
    for i, el in enumerate(k):
        if el == 0:
            z_0 += 1
            z_1 = 0
            if z_0 >= m_0:
                m_0 = z_0
            max_st_nicel[i] = m_0
            max_st_enic[i] = m_1

        else:
            z_1 += 1
            z_0 = 0
            if z_1 >= m_1:
                m_1 = z_1
            max_st_enic[i] = m_1
            max_st_nicel[i] = m_0
            
    # Graf 2
    fig3 = plt.figure()
    ax3 = fig3.gca()
    ax3.set(title= "Spreminjanje maksimane dolžine zaporednih enic in ničel", xlabel = "korak - t", ylabel = "zaporedno število ponovitve")
    ax3.scatter(st_korakov[:], max_st_enic[:], c="blue",marker = ".", label = "Enice")
    ax3.scatter(st_korakov[:], max_st_nicel[:], c="violet",marker = ".", label = "Ničle")
    plt.savefig("gruce.pdf")
    
    handles, labels = ax3.get_legend_handles_labels()
    ax3.legend(handles, labels, fontsize = 13)
 
    # Graf 1
    pov = (np.array(st_enic) + np.array(st_nicel))/2
    fig = plt.figure()
    ax = fig.gca()
    ax.set(title = "Število enic in ničel ob nekem koraku", xlabel = "korak - t", ylabel = "število - n")
    ax.plot(st_korakov[:], st_enic[:], c="blue", label="Število enic")
    ax.plot(st_korakov[:], st_nicel[:], c="violet", label = "Število ničel")
    ax.plot(st_korakov[:], pov[:], c = "grey", label = "Premica")
    plt.savefig("stat.pdf")
    
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize = 13)

drugi_del(s_z_2, pravilo_1, t_2)

plt.show()
        


  
