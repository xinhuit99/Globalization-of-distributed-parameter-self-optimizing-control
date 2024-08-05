# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 18:50:20 2024

@author: tangXinHui
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 21:33:31 2023

@author: tangXinHui
"""


import casadi as ci
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib import cm
import pickle
import os
from scipy.linalg import block_diag
import copy
import pandas as pd

path = r'D:/anaconda2021/directorySpyder/academic/second paper'

os.chdir(path)

# In[] model (the arguments related to C are not used)

# reaction parameter
L = 1 # reactor length
N = 90 # separate region number
numu = 3 # number of control input 
numym = 9+1 # number of measuring points

# model variable
T = ci.SX.sym('T') 
CA = ci.SX.sym('CA')
CC = ci.SX.sym('CC')

# input
Tw = ci.SX.sym('Tw')

# boundary variable
T0 = ci.SX.sym('T0')
CA0 = ci.SX.sym('CA0')
CC0 = ci.SX.sym('CC0')

# parameter variable
v = ci.SX.sym('v')
EA = ci.SX.sym('EA')
R = ci.SX.sym('R')
kA = ci.SX.sym('kA')
khC = ci.SX.sym('khC')
HA = ci.SX.sym('HA')
HC = ci.SX.sym('HC')
h = ci.SX.sym('h')
klC = ci.SX.sym('klC')

y = ci.vertcat(T, CA, CC)
y0 = ci.vertcat(T0, CA0, CC0)
u = Tw
d = ci.vertcat(v, EA, R, kA, khC, HA, HC, h, klC)

# nominal value
# model boundary
T0_nominal = 340
CA0_nominal = 0.02
CC0_nominal = 0.0

# model parameter
v_nominal = 0.1
EA_nominal = 11250
R_nominal = 1.986
kA_nominal = 10**6
khC_nominal = 0.5*kA_nominal*np.exp(-EA_nominal/R_nominal/T0_nominal)
HA_nominal = 0.25*T0_nominal/CA0_nominal # -HA/rou/Cp
HC_nominal = 0.8*0.25*T0_nominal/CA0_nominal # -HC/rou/Cp
h_nominal = 0.2 # 4h/rou/Cp/d
klC_nominal = 0.1

d_nominal = [T0_nominal, CA0_nominal, CC0_nominal, 
             v_nominal, EA_nominal, R_nominal, kA_nominal, khC_nominal, HA_nominal, HC_nominal, h_nominal, klC_nominal]


# model function
ydot = ci.vertcat( 
    1/v*( HA*kA*np.exp(-EA/R/T)*CA + h*(Tw-T) + HC*khC*CC ), 
    1/v*( -kA*np.exp(-EA/R/T)*CA ),
    1/v*( -khC*CC-klC*CC**2)
                             )

# In[] spatial discretization by Finite difference

# integrated objective term
Li = 0 

# construct function
if False:
# direct integral
    dae = {'x': y, 'p': u, 'ode': ydot, 'quad': Li}
    F = ci.integrator('F', 'cvodes', dae, 0, L/N)
else:
    # Runge-Kutta 4 integrator
    M = 8 # RK4 steps per interval
    DT = L/N/M
    f = ci.Function('f', [y,u,d], [ydot, Li])
    Y0 = ci.SX.sym('Y0',3)
    Ds = ci.SX.sym('Ds',9)
    U = ci.SX.sym('U')
    Y = Y0
    Q = 0
    for j in range(M):
        k_1, k1_q = f(Y, U, Ds)
        k_2, k2_q = f(Y + DT/2*k_1, U, Ds)
        k_3, k3_q = f(Y + DT/2*k_2, U, Ds)
        k_4, k4_q = f(Y + DT*k_3, U, Ds)
        Y = Y + DT/6*(k_1 + 2*k_2 + 2*k_3 + k_4)
        Q = Q + DT/6*(k1_q +2*k2_q +2*k3_q + k4_q)
    F = ci.Function('F', [Y0, U, Ds], [Y, Q], ['y0','p','para'],['yf','qf'])
    
    
# In[] optimization problem

yk_var = ci.vertcat(T0, CA0, CC0)
ym_var = [yk_var[0]] # inlet temperature is available
ystate_var = [yk_var]
u_var = []
lbx = []
ubx = []
lbg = []
ubg = []

g_var = []
J_var = 0
phi_var = 0

for k in range(N):
    if k%(N/numu) == 0:
        uk_var = ci.SX.sym('u_'+str(int(k/(N/numu))))
        u_var.append(uk_var)
        lbx.append(-np.inf)
        ubx.append(np.inf)
    
    Fk_var = F(y0=yk_var, p=uk_var, para=d)
    
    yk_var = Fk_var['yf']
    J_var = J_var + Fk_var['qf']  # integral cost
    
    ystate_var.append(yk_var)
    
    if (k+1)%(N/(numym-1)) == 0:
        ym_var.append(yk_var[0])

    # target
    phi_var = phi_var + (yk_var[0]-T0)**2
    
ym_var = ci.vcat(ym_var)
ystate_var = ci.vcat(ystate_var)
u_var = ci.vcat(u_var)
d_var = ci.vertcat(y0, d)

# optimization problem
nlp_var = {'x': u_var,'f': phi_var, 'p': d_var, 'g': g_var} 
solver = ci.nlpsol('nlp','ipopt', nlp_var)

# sensitivity function
Gy_var = ci.jacobian(ym_var, u_var)
[Juu_var, Ju_var] = ci.hessian(phi_var, u_var)

# output function
Output_func = ci.Function('Output_func', [u_var, d_var], [ym_var, ystate_var, Gy_var, Juu_var, Ju_var], ['p','para'], ['ymf','ystatef','Gyf','Juuf', 'Juf'])

# In[] nominal solution 

# disturbance value
p_temp = d_nominal

# solution
sol_temp = solver(
    x0 = T0_nominal*np.ones(numu),
    p = p_temp,
    lbx = lbx,
    ubx = ubx,
    lbg = lbg,
    ubg = ubg
             )

# data type transformation
u_temp = sol_temp['x'].full()[:,0]
phi_temp = sol_temp['f'].full()[:,0]

output_temp = Output_func(p=u_temp, para=p_temp)

ym_temp = output_temp['ymf'].full()
ystate_temp = output_temp['ystatef'].full()
Gy_temp = output_temp['Gyf'].full()
Juu_temp = output_temp['Juuf'].full()
Ju_temp = output_temp['Juf'].full()

# In[] result visulization

ystate_T_temp = ystate_temp[::3,0]
ystate_CA_temp = ystate_temp[1::3,0]
ystate_CC_temp = ystate_temp[2::3,0]
z_plot = np.arange(0,L+L/N,L/N)

plt.figure()

plt.xlabel('$\it{z}$', fontsize=14)
plt.ylabel('$\it{T} (K)$', fontsize=14)
plt.plot(z_plot, ystate_T_temp)

plt.figure()
plt.xlabel('$\it{z}$', fontsize=14)
plt.ylabel('$\it{C}_\mathrm{A}$ (mol/L)', fontsize=14)
plt.plot(z_plot, ystate_CA_temp)

plt.figure()
plt.xlabel('$\it{z}$', fontsize=14)
plt.ylabel('$\it{T}_\mathrm{w}$ (K)', fontsize=14)
plt.hlines(u_temp[0], xmin=0, xmax=1/3)
plt.hlines(u_temp[1], xmin=1/3, xmax=2/3)
plt.hlines(u_temp[2], xmin=2/3, xmax=1)
plt.vlines(x=1/3, ymin=u_temp[0], ymax=u_temp[1] )
plt.vlines(x=2/3, ymin=u_temp[1], ymax=u_temp[2] )

# In[] addition sensitivity matrix 

# additional sensitivity matrix
Gydall_var = ci.jacobian(ym_var, ci.vertcat(T0, CA0, v))
Judall_var = ci.jacobian(Ju_var, ci.vertcat(T0, CA0, v))

# output function
Output_linear_func = ci.Function('Output_linear_func', [u_var, d_var], [ym_var, ystate_var, Gy_var, Juu_var, Ju_var, Gydall_var, Judall_var], 
                              ['p','para'], ['ymf','ystatef','Gyf','Juuf', 'Juf', 'Gydallf', 'Judallf'])


# In[] nominal solution and related matrix

# disturbance value
p_temp = d_nominal

# solution
sol_temp = solver(
    x0 = T0_nominal*np.ones(numu),
    p = p_temp,
    lbx = lbx,
    ubx = ubx,
    lbg = lbg,
    ubg = ubg
             )

# data type transformation
u_temp = sol_temp['x'].full()[:,0]
phi_temp = sol_temp['f'].full()[:,0]

output_temp = Output_linear_func(p=u_temp, para=p_temp)

ym_temp = output_temp['ymf'].full()
ystate_temp = output_temp['ystatef'].full()
Gy_temp = output_temp['Gyf'].full()
Juu_temp = output_temp['Juuf'].full()
Ju_temp = output_temp['Juf'].full()
Gydall_temp = output_temp['Gydallf'].full()
Judall_temp = output_temp['Judallf'].full()

ym_nominal_sample = ym_temp
ystate_nominal_sample = ystate_temp
u_nominal_sample = u_temp
Gy_nominal_sample = Gy_temp
Juu_nominal_sample = Juu_temp
Gydall_nominal_sample = Gydall_temp
Judall_nominal_sample = Judall_temp


# In[] disturbance and noise variance (uniform distribution)

# type 1: v 0.03, T0 1, CA0 0.01 small
# type 2: v 0.022, T0 1, CA0 0.01 small
disturbance_type = '1'
v_range_sample = 0.03
T0_range_sample = 1
CA0_range_sample = 0.01

# noise matrix
noise_range = 0.02


# In[] local method: solve linear H
    
V_nominal_sample = np.rot90(np.linalg.cholesky(np.rot90(Juu_nominal_sample, 2)).T , 2)
F_nominal_sample = -Gy_nominal_sample.dot( 
    np.linalg.inv(Juu_nominal_sample).dot(Judall_nominal_sample)
    ) + Gydall_nominal_sample

Wn_nominal_sample = np.diag( noise_range*np.ones(numym) )
Wd_nominal_sample = np.diag( [T0_range_sample, CA0_range_sample, v_range_sample] )

Fbar_nominal_sample = np.hstack((
    F_nominal_sample.dot( Wd_nominal_sample ),
    Wn_nominal_sample
    ))

Hl_sample = np.linalg.inv(Fbar_nominal_sample.dot(Fbar_nominal_sample.T)).dot(
    Gy_nominal_sample
    ).dot(
        np.linalg.inv(
            Gy_nominal_sample.T.dot(
                np.linalg.inv(Fbar_nominal_sample.dot(Fbar_nominal_sample.T))
                ).dot(Gy_nominal_sample)
            )
        ).dot(V_nominal_sample)

Hl_sample = Hl_sample.T


# In[] global method: # sampling settings (disturbance number, sample number)

sample_var = 3
sample_num = 40

# In[] global method: training samples

# random sample
d_random_sample = np.hstack(
     (
     np.random.normal(v_nominal, v_range_sample, size = (sample_num, 1)),
     np.random.normal(T0_nominal, T0_range_sample, size = (sample_num, 1)),
     np.random.normal(CA0_nominal, CA0_range_sample, size = (sample_num, 1))
     )
    )

d_random_statistic_sample = [ np.mean(d_random_sample, axis = 0), np.var(d_random_sample, axis = 0) ]


ymopt_random_sample = []
uopt_random_sample = []
Gy_random_sample = []
Juu_random_sample = []

ymopt_lhs_sample = []
uopt_lhs_sample = []
Gy_lhs_sample = []
Juu_lhs_sample = []


p_temp = copy.deepcopy(d_nominal)

# random sampling 
for i in range(sample_num):
    
    # disturbance value
    p_temp[0] = d_random_sample[i,1]
    p_temp[1] = d_random_sample[i,2]
    p_temp[3] = d_random_sample[i,0]
    
    

    # solution
    sol_temp = solver(
        x0 = T0_nominal*np.ones(numu),
        p = p_temp,
        lbx = lbx,
        ubx = ubx,
        lbg = lbg,
        ubg = ubg
                 )
    
    if solver.stats()['return_status'] == 'Solve_Succeeded':
        # data type transformation
        u_temp = sol_temp['x'].full()[:,0]
        phi_temp = sol_temp['f'].full()[:,0]
    
        output_temp = Output_func(p=u_temp, para=p_temp)
    
        ym_temp = np.vstack((np.ones((1,1)), 
                             output_temp['ymf'].full() 
                             ))
        ystate_temp = output_temp['ystatef'].full()
        Gy_temp = np.vstack((np.zeros((1,numu)) ,
            output_temp['Gyf'].full()
            ))
        Juu_temp = output_temp['Juuf'].full()
        Ju_temp = output_temp['Juf'].full()
        
        # collect data for random sampling
        ymopt_random_sample.append(ym_temp)
        uopt_random_sample.append(u_temp)
        Gy_random_sample.append(Gy_temp)
        Juu_random_sample.append(Juu_temp)
        
    else:
        continue
        
    

# noise matrix with uniform distribution (-0.02,0.02)
W2_sample = noise_range**2*np.diag( np.concatenate(( np.zeros((1)),np.ones((numym)) )), k = 0)


# In[] global method: optimization problem of solving global H 
Hg_random = ci.SX.sym('Hg_random', numu, numym+1)

phi_H_random = 0 
for i in range(len(ymopt_random_sample)):
    phi_H_random = phi_H_random + \
        ci.mtimes(ymopt_random_sample[i].T, 
               ci.mtimes(Hg_random.T,
                      ci.mtimes(ci.inv( ci.mtimes(Hg_random, Gy_random_sample[i]) ).T,
                             ci.mtimes( Juu_random_sample[i],
                                    ci.mtimes(ci.inv( ci.mtimes(Hg_random, Gy_random_sample[i]) ),
                                           ci.mtimes(Hg_random, 
                                                  ymopt_random_sample[i])
                                        )
                                 )
                          )
                      )
               ) + \
        ci.trace(ci.mtimes(W2_sample,
                     ci.mtimes(Hg_random.T,
                            ci.mtimes(ci.inv( ci.mtimes(Hg_random, Gy_random_sample[i]) ).T,
                                   ci.mtimes( Juu_random_sample[i],
                                          ci.mtimes(ci.inv( ci.mtimes(Hg_random, Gy_random_sample[i]) ),
                                                 ci.mtimes(ci.inv( ci.mtimes(Hg_random, Gy_random_sample[i]) ),
                                                        Hg_random)
                                              )
                                        )
                                )
                         )
                  )
            )
            

# optimization problem
nlp_Hg_random_var = {'x': ci.reshape(Hg_random,(-1,1)), 
                         'f': phi_H_random, 
                         'g': ci.reshape( ci.mtimes(Hg_random, np.vstack(( np.zeros((1,3)),Gy_nominal_sample ))) - np.rot90(np.linalg.cholesky(np.rot90(Juu_nominal_sample,2)).T, 2), (-1,1) ) } 
solver_Hg_random = ci.nlpsol('nlp','ipopt', nlp_Hg_random_var)

# In[] global method: solving global H 

sol_Hg_random_temp = solver_Hg_random(
    x0 = np.vstack((
        (-Hl_sample.dot(ym_nominal_sample)).reshape(-1,1), Hl_sample.T.reshape(-1,1, order = 'f')
        )),
    lbg = np.zeros(numu*numu),
    ubg = np.zeros(numu*numu)
             )

# value
Hg_random = sol_Hg_random_temp['x'].full().reshape(numu, numym+1, order = 'f')

# In[] test: optimization strategies with different SOC CVs

# testing sample number
testing_num = 200

# noise variables
noise_var = ci.SX.sym('noise_var', numym, 1)
yr_var = ym_var + noise_var

# cv value and optimization problem of LPS
cv_lps_var = ci.mtimes( block_diag(np.ones((1,3)), np.ones((1,3)), np.ones((1,3))), yr_var[1:])/3 - yr_var[0]
phi_lps_var = ci.mtimes(cv_lps_var.T, cv_lps_var)

nlp_lps_var = {'x': u_var, 'f': phi_lps_var, 'p': ci.vcat([d_var, noise_var]), 'g': g_var}
solver_lps = ci.nlpsol('nlp', 'ipopt', nlp_lps_var)

# cv value and optimization problem of linear H
cv_linear_var = ci.mtimes(Hl_sample, (yr_var-ym_nominal_sample))
phi_linear_var = ci.mtimes(cv_linear_var.T, cv_linear_var)

nlp_linear_var = {'x': u_var, 'f': phi_linear_var, 'p': ci.vcat([d_var, noise_var]), 'g': g_var}
solver_linear = ci.nlpsol('nlp', 'ipopt', nlp_linear_var)

# cv value and optimization problem of global H for random sampling
cv_global_random_var = ci.mtimes(Hg_random, ci.vertcat(1,yr_var))
phi_global_random_var = ci.mtimes(cv_global_random_var.T, cv_global_random_var)

nlp_global_random_var = {'x': u_var, 'f': phi_global_random_var, 'p': ci.vcat([d_var, noise_var]), 'g': g_var}
solver_global_random = ci.nlpsol('nlp', 'ipopt', nlp_global_random_var)

# testing output function
Output_testing_func = ci.Function('Output_testing_func', [u_var, ci.vcat([d_var, noise_var])], [phi_var, ym_var, yr_var, ystate_var], 
                              ['p','para'], ['phif','ymf','yrf','ystatef'])

# In[] test: disturbance and noise

# testing disturbance
d_random_testing = np.hstack(
     (
     np.random.normal(v_nominal, v_range_sample, size = (testing_num, 1)),
     np.random.normal(T0_nominal, T0_range_sample, size = (testing_num, 1)),
     np.random.normal(CA0_nominal, CA0_range_sample, size = (testing_num, 1))
     )
    )

d_random_statistic_testing = [ np.mean(d_random_testing, axis = 0), np.var(d_random_testing, axis = 0) ]

# testing noise
noise_random_testing = np.random.normal(0, noise_range, size = (testing_num, numym))

noise_random_statistic_testing = [ np.mean(noise_random_testing, axis = 0), np.var(noise_random_testing, axis = 0) ]

# In[] testing (remove abnormal values)

# changeable parameters ([T0, CA0, CC0, v, EA, R, kA, khC, HA, HC, h, klC])
ymopt_testing = []
ystateopt_testing = []
uopt_testing = []
phiopt_testing = []

ym_lps_testing = []
yr_lps_testing = []
ystate_lps_testing = []
u_lps_testing = []
phi_lps_testing = []

ym_linear_testing = []
yr_linear_testing = []
ystate_linear_testing = []
u_linear_testing = []
phi_linear_testing = []

ym_global_random_testing = []
yr_global_random_testing = []
ystate_global_random_testing = []
u_global_random_testing = []
phi_global_random_testing = []


for i in range(testing_num):
    
    # random sampling 
    
    # disturbance value
    p_temp = copy.deepcopy(d_nominal)
    p_temp[0] = d_random_testing[i,1]
    p_temp[1] = d_random_testing[i,2]
    p_temp[3] = d_random_testing[i,0]
    
    
    
    # solution
    # for optimal
    solopt_temp = solver(
        x0 = T0_nominal*np.ones(numu),
        p = p_temp,
        lbx = lbx,
        ubx = ubx,
        lbg = lbg,
        ubg = ubg
                 )
    
    # data type transformation
    u_temp = solopt_temp['x'].full()[:,0]
    phi_temp = solopt_temp['f'].full()[:,0]

    output_temp = Output_func(p=u_temp, para=p_temp)

    ym_temp = output_temp['ymf'].full() 
                         
    ystate_temp = output_temp['ystatef'].full()
    
    
    
    # noise value
    p_temp.extend(noise_random_testing[i,:])
    
    
    
    # for lps
    sol_lps_temp = solver_lps(
        x0 = T0_nominal*np.ones(numu),
        p = p_temp,
        lbx = lbx,
        ubx = ubx,
        lbg = lbg,
        ubg = ubg
                 )
    # for linear
    sol_linear_temp = solver_linear(
        x0 = T0_nominal*np.ones(numu),
        p = p_temp,
        lbx = lbx,
        ubx = ubx,
        lbg = lbg,
        ubg = ubg
                 )
    
    # for global random
    sol_global_random_temp = solver_global_random(
        x0 = T0_nominal*np.ones(numu),
        p = p_temp,
        lbx = lbx,
        ubx = ubx,
        lbg = lbg,
        ubg = ubg
                 )
    
    
    
    if solver.stats()['return_status'] == 'Solve_Succeeded' and solver_lps.stats()['return_status'] == 'Solve_Succeeded' and solver_linear.stats()['return_status'] == 'Solve_Succeeded' and solver_global_random.stats()['return_status'] == 'Solve_Succeeded':
        
        # collect data for optimal
        ymopt_testing.append(ym_temp)
        ystateopt_testing.append(ystate_temp)
        uopt_testing.append(u_temp)
        phiopt_testing.append(phi_temp)
    
        
        
        # data type transformation
        u_temp = sol_lps_temp['x'].full()[:,0]
        phi_temp = sol_lps_temp['f'].full()[:,0]
    
        output_temp = Output_testing_func(p=u_temp, para=p_temp)
    
        ym_temp = output_temp['ymf'].full() 
        
        yr_temp = output_temp['yrf'].full() 
                             
        ystate_temp = output_temp['ystatef'].full()
    
        phif_temp = output_temp['phif'].full()
        
        # collect data for lps
        ym_lps_testing.append(ym_temp)
        yr_lps_testing.append(yr_temp)
        ystate_lps_testing.append(ystate_temp)
        u_lps_testing.append(u_temp)
        phi_lps_testing.append(phif_temp)
        
        
    
        # data type transformation
        u_temp = sol_linear_temp['x'].full()[:,0]
        phi_temp = sol_linear_temp['f'].full()[:,0]
    
        output_temp = Output_testing_func(p=u_temp, para=p_temp)
    
        ym_temp = output_temp['ymf'].full()
        
        yr_temp = output_temp['yrf'].full() 
                             
        ystate_temp = output_temp['ystatef'].full()
    
        phif_temp = output_temp['phif'].full()
        
        # collect data for linear
        ym_linear_testing.append(ym_temp)
        yr_linear_testing.append(yr_temp)
        ystate_linear_testing.append(ystate_temp)
        u_linear_testing.append(u_temp)
        phi_linear_testing.append(phif_temp)
        
        
    
        # data type transformation
        u_temp = sol_global_random_temp['x'].full()[:,0]
        phi_temp = sol_global_random_temp['f'].full()[:,0]
    
        output_temp = Output_testing_func(p=u_temp, para=p_temp)
    
        ym_temp = output_temp['ymf'].full()
        
        yr_temp = output_temp['yrf'].full() 
                             
        ystate_temp = output_temp['ystatef'].full()
    
        phif_temp = output_temp['phif'].full()
        
        # collect data for global random sampling
        ym_global_random_testing.append(ym_temp)
        yr_global_random_testing.append(yr_temp)
        ystate_global_random_testing.append(ystate_temp)
        u_global_random_testing.append(u_temp)
        phi_global_random_testing.append(phif_temp)
        
        
    else:
        continue
        
    
# In[] statistic result of testting

meanphiopt = np.mean(np.array(phiopt_testing)[:,0])
print('optimal mean: '+str(meanphiopt))


meanphi_lps_testing = np.mean(np.array(phi_lps_testing)[:,0,0])
print('lps mean: '+str(meanphi_lps_testing))

meanphi_linear_testing = np.mean(np.array(phi_linear_testing)[:,0,0])
print('linear mean: '+str(meanphi_linear_testing))

meanphi_global_random_testing = np.mean(np.array(phi_global_random_testing)[:,0,0])
print('global random mean: '+str(meanphi_global_random_testing))


print('------')

meanloss_lps_testing = meanphi_lps_testing - meanphiopt
print('lps mean loss: '+str(meanloss_lps_testing))
meanloss_linear_testing = meanphi_linear_testing - meanphiopt
print('linear mean loss: '+str(meanloss_linear_testing))
meanloss_global_random_testing = meanphi_global_random_testing - meanphiopt
print('global random mean loss: '+str(meanloss_global_random_testing))

print('------')

maxloss_lps_testing = np.max(np.array(phi_lps_testing)[:,0,0]-np.array(phiopt_testing)[:,0])
print('lps max loss: '+str(maxloss_lps_testing))
maxloss_linear_testing = np.max(np.array(phi_linear_testing)[:,0,0]-np.array(phiopt_testing)[:,0])
print('linear max loss: '+str(maxloss_linear_testing))
maxloss_global_random_testing = np.max(np.array(phi_global_random_testing)[:,0,0]-np.array(phiopt_testing)[:,0])
print('global random max loss: '+str(maxloss_global_random_testing))