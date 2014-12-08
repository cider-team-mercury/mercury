'''
Solve Two Layer Conduction with heating
'''
__author__ = 'bdel'


import numpy as np
from numpy import power
import matplotlib
matplotlib.use('macosx')
import matplotlib.pyplot as plt

kc = 2.0
km = 4.0
qc = 1.e-14
qm  = 1.e-14
rs = 2440
rl = 200
rc = 1320
Ts = 500
Tl = 1800
kp =kc/km
delta_q = qc-qm
dT = Tl - Ts
k=kc/km

alpha = power(rc, 3.)*delta_q/(3*km)
beta = power(rc, 2)*(qm/km -qc/kc)/6 -alpha/rc
gamma = -(qm*power(rl,2.))/(6*km) + alpha/rl + beta

top = -qc*power(rs,2)/(6*kc)-gamma +dT
bottom = k/rl +(1-k)/rc -1/rs

c1 = top/bottom
c2 = Tl -gamma - (k/rl +(1-k)/rc)*c1
m1 = alpha + k*c1
m2 = beta + (1-k)*c1/rc +c2

crust_solution  = lambda r: (-qc*r*r/(6.*kc) + c1/r + c2) if((rs  >= r)and(r >=rc))  else 0
mantle_solution = lambda r: (-qm*r*r/(6.*km) + m1/r + m2) if((rc >= r)and(r >= rl))  else 0
solution = lambda r: crust_solution(r) + mantle_solution(r)


r_sol = np.linspace(rl,rs,1000)
sol = np.empty_like(r_sol)
for ii, r in enumerate(r_sol):
    sol[ii] = solution(r)
plt.plot(r_sol, sol)
plt.show()



