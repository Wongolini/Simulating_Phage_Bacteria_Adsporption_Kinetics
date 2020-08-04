#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
%%prun
import scipy as sp
import matplotlib.pyplot as plt
import math 
import timeit

#We implement gillespie with 3e+8 bacteria
#To accomodate for the scale, we include tau leaping method
#We compare explicit and implicit tau leaping methods
#We will test this on small and large MOIs
#Compare results to Moldevan et al. 2007 ODEs

def Generate_Population(num_bacteria, max_receptors):
    #Generates initial population and bacterial species with max M receptors
    #pop is the composition vector X
    pop = np.zeros((num_bacteria, 3),dtype=np.int32)
    pop[:,0] = np.random.randint(low=1, high = max_receptors,
                                size = (num_bacteria))
    return pop

def Generate_Data(pop):
    M = np.max(pop[:,0])
    data = np.zeros((M,M),dtype=np.int32)
    return data

def Vectorization(num_bacteria, num_phages, k1,k2,k3, pop):
    #Propensity Function: SUM(a_j(x)) where a_j is the propensity 
    # or intensity for reaction X_j given by reaction kinetic scheme
    #We say that the system evolves after a leap tau following a 
    #Poisson dist with mean and variance SUM(a_j(x))*tau
    rxn_vector = np.zeros((num_bacteria,3),dtype=np.float32)
    rxn_vector[:,0] = k1*pop[:,0]*num_phages #adsorb    
    rxn_vector[:,1] = k2*pop[:,1]            #desorb
    rxn_vector[:,2] = k3*pop[:,1]            #perm
    return rxn_vector

def update_data(pop,data):
    count_pop = pop[:,1:3]
    unique, counts = np.unique(count_pop, axis = 0, return_counts = True)
    #x axis of data is nonperm
    #y axis of data is perm
    for i in range(len(counts)):
        x = unique[i][0]
        y = unique[i][1]
        data[x][y] = counts[i]
    return data

def SSA(pop, num_bacteria, num_phages):
    #Simulate the transition type by drawing from the discrete distribution
    #with probability Prob(trans = i) = lambda_i(x)/lambda
    #Generate a tarndom number from a uniform dist and choose the
    #transition as follows
    #if 0<r<lambda1/lambda, choose transition 1

    #### Current Issue:  Does not account for negative populations.  For some reason adsorption reactions
    #### Can still occur even if Phage populatiosn = 0 or receptor status = 0.
    #### The impossibility of these reactions should be contained in the formula?
    #### Defeats the point of SSA since SSA is used to prevent negative populations
    #### Check the math.
    start = timeit.default_timer()
    t = 0
    for i in range(100):
        rxn_vector = Vectorization(num_bacteria,num_phages,k1,k2,k3,pop)
        tot_prop = np.sum(rxn_vector)
        print('iteration', i)
        print('num phages', num_phages)
        aj_prime = 0
        tau = (1/tot_prop)*np.log(1/np.random.uniform())

        #index of the next rxn is the smallest integer j satsifying: SUM_j'^j[(aj'(x))>r*a_0(x)]
        upper_limit = np.random.uniform()*tot_prop #issue with upper limit condition
        #if the upperlimit = 0, then that means no more reactions can occur and the program must stop.
        if upper_limit == 0:
            break
        
        for x in range(num_bacteria):   #parse through bacteria
            for c_, y in enumerate(rxn_vector[x]):     #parse through receptor status
                aj_prime += y
                if aj_prime > upper_limit:
                    break
            if aj_prime > upper_limit:
                break
        print('c',c_,y)
        print('pop before change', pop[x])
        print('rxn vector' , rxn_vector[x])
        if c_ == 0: #adsorb 1 phage
            pop[x] += [-1,+1,0]
            num_phages -= 1
                    
        if c_ == 1: #desorb 1 phage
            pop[x] += [+1,-1,0]
            num_phages += 1

        if c_ == 2: #perm phage
            pop[x] += [0,-1,+1]
        #checking to see why we get negative populations
        #weird things happen when rxn_vector[x] == [0,0,0]
        if num_phages<0:
            #plt.yscale('log')
            print('ajprime',aj_prime)
            plt.scatter([1]*num_bacteria,rxn_vector[:,0])
            plt.scatter([2]*num_bacteria,rxn_vector[:,1])
            plt.scatter([3]*num_bacteria,rxn_vector[:,2])
            plt.show()
            import sys
            sys.exit()
        print('pop after change' , pop[x])
        t += tau
    stop = timeit.default_timer()
    print('Time SSA: ', stop - start) 
    print('num phages', num_phages)
    return t,pop,num_phages

def Evolution(num_bacteria, num_phages, pop, rxn_vector):
    #K_j(tau,x,t) = number of times, given X(t)=x that a rxn channel R_j
    #will fire in the time interval [t,t+tau], j=1,...n
    #K_j = Poiss(aj(x)*tau)

    #X(t+tau) = x+SUM(Kj*vj), vj is the change of state vector
    #We must calculate a tau large enought to allow for several reactions,
    #but small enough to not generate unrealistic results.
    #Given X(t) = x, identify all critical transitions by 
    # estimating the maximum number of times, Lj, a transition can
    # occur before causing a negative population

    #### I some changes to this formulation
    #### Rather than critical transitions we consider critical populations
    #### Subset the bacteria and their receptor status based on the Lj condition
    #Lj = min[xi/|vij|]<--pop
    #### We do this because some populations are so small that there will always exist a critical transition
    #### this is like treating bacteria as individual molecules now
    nc = 10   #nc is chosen heuristically after many trials
    #A transition is considered critical if Lj<nc
    #find the critical and noncritical transitions
    #each entry in pop has a state of change vector 
    v1 = np.array([-1,-1,1,0])  #-P -Xi1 +Xi2  0
    v2 = np.array([1,1,-1,0])   #+P +Xi1 -Xi2  0
    v3 = np.array([0,0,-1,1])   #0   0   -Xi2 +Xi3
    #vj for j = 1,2,3 transitions and ith entry says what molecular species is changinng
    vj = np.array([v1,v2,v3])
    #V = np.array(np.where(vj<0)) #the indices of vij<0
    #3 reactions even though we have many species
    tot_prop = np.sum(rxn_vector)

    X1 = pop[:,0] #set of free receptors among all bacteria
    #remember that P is also population that can be critical
    X2 = pop[:,1] #set of adhered receptors among all bacteria
    X1_prime = X1[X1>nc]#subset of free receptors that are not critical
    X1_prime_ind = np.array(np.where(X1>nc)) #indices of free receptors not critical
    X1_prime_ind = X1_prime_ind.flatten()
    X2_prime = X2[X2>nc]#subset of adhered receptors that are not critical
    X2_prime_ind = np.array(np.where(X2>nc)) #indices of adhered receptors not critical
    X2_prime_ind = X2_prime_ind.flatten()
    #we calculate the rate of change for propensities
    
    fjj = np.zeros(3)
    uj = np.zeros(3)
    sigmaj = np.zeros(3)

    if (len(X1_prime) == 0 or num_phages<nc) and len(X2_prime) == 0 :
        #initiate SSA <-- turn this into a seperate definition since there are two contingencies this can happen
        t, pop, num_phages = SSA(pop, num_bacteria, num_phages)
        
    else:
        fjj[0] = -np.sum(X1)  *k1 + -num_phages * k1
        fjj[1] = len(X2) * k2
        fjj[2] = -len(X2) * k3
        
        X1_prime_sum = np.sum(X1_prime)
        X2_prime_sum = np.sum(X2_prime)

        uj[0] = fjj[0] * X1_prime_sum * num_phages * k1
        uj[1] = fjj[1] * X2_prime_sum * k2
        uj[2] = fjj[2] * X2_prime_sum * k3

        sigmaj[0] = fjj[0]**2 * X1_prime_sum * num_phages * k1
        sigmaj[1] = fjj[1]**2 * X2_prime_sum * k2
        sigmaj[2] = fjj[2]**2 * X2_prime_sum * k3

    #calculate a tau so each transition rate (propensity) is bounded above by error parameter e
        e = 0.03
        tau1 = np.zeros((3,2))
        
        tau1[0] = np.array([e*tot_prop/abs(uj[0]), e*np.power(tot_prop,2)/np.power(sigmaj[0],2)])
        tau1[1] = np.array([e*tot_prop/abs(uj[1]), e*np.power(tot_prop,2)/np.power(sigmaj[1],2)])
        tau1[2] = np.array([e*tot_prop/abs(uj[2]), e*np.power(tot_prop,2)/np.power(sigmaj[2],2)])
        tau1 = np.nanmin(tau1)
        
        #mutate change in time, and record evolution

        if (tau1 < 0.1*1/tot_prop):
            #initiate SSA instead 100 times
            t, pop, num_phages = SSA(pop, num_bacteria, num_phages)
            print('time in seconds', t)
        else:
            #generate a second tau exponentially distributed from the critical transitions
            X1_double_prime = X1[X1<nc]
            X2_double_prime = X2[X2<nc]
            X1_double_prime_ind = np.array(np.where(X1<nc)).flatten()
            X2_double_prime_ind = np.array(np.where(X2<nc)).flatten()
            rate = np.sum(X1_double_prime)*k1*num_phages+np.sum(X2_double_prime)*k2+np.sum(X2_double_prime)*k3
            print('exp rate', rate)
            tau2 = np.random.exponential(1/(rate)) 
            #tau2 is the expected time to the next critical reaction
            print('tau1', tau1)
            print('tau2', tau2)
            
            #If we our tau leap is safe, then assume .transitions can leap with poisson r.v.
            #if tau2<=tau1 then tau = tau2.  
            # Generate jc as a samle of the integer random variable with point
            # probabilities aj(x)/sum(aj_critical(x)) where j runs over the index
            # values of the critical reactions only -- the only critical reaction
            # that will fire in this leap
            # Set kjc = 1 and for all other critical reactions Rj, set kj = 0
            # For all noncritical reactions Rj, generate kj as a sample of
            # the poisson random variable with mean aj(x)*tau

            tau = np.nanmin([tau1,tau2])
            print('tau', tau)
            #tau = tau1 --> no critical reactions will fire
            #tau = tau2 --> choose only one critical reaction kj=1 and all other kj=0
            if tau == tau1:
                Kj = np.random.poisson(rxn_vector*tau) #generates number of transitions that can occur for each bacteria and receptor status
            else:
                #Generate point probabilities for critical reactions and choose one
                print('======CRITICAL REACTIONS DETECTED======')
                criticalX1 = np.array(rxn_vector[:,0][X1_double_prime_ind])
                criticalX2 = np.array(rxn_vector[:,1][X2_double_prime_ind])
                criticalX3 = np.array(rxn_vector[:,2][X2_double_prime_ind])
                print('Number of Criticals:' , len(criticalX1)+len(criticalX2) + len(criticalX3))
                criticalX1_sum = np.sum(criticalX1)
                criticalX2_sum = np.sum(criticalX2)
                criticalX3_sum = np.sum(criticalX3)
                critical_sum = criticalX1_sum + criticalX2_sum + criticalX3_sum
                
                # Normalize
                criticalX1 = criticalX1/critical_sum
                criticalX2 = criticalX2/critical_sum
                criticalX3 = criticalX3/critical_sum
                criticalX1_sum /= critical_sum
                criticalX2_sum /= critical_sum
                criticalX3_sum /= critical_sum
                
                choices = []
                p_choices = []
                
                if len(criticalX1) == 0 or np.all(criticalX1 == 0):
                    a = np.NaN
                    choices.append(a)
                    p_choices.append(0)
                else:
                    a = np.random.choice(X1_double_prime_ind, size = 1, p = criticalX1/criticalX1_sum)
                    choices.append(a[0])
                    p_choices.append(X1[a[0]])
                
                if len(criticalX2) == 0 or np.all(criticalX2 == 0):
                    b = np.NaN
                    choices.append(b)
                    p_choices.append(0)
                else:
                    b = np.random.choice(X2_double_prime_ind, size = 1, p = criticalX2/criticalX2_sum)
                    choices.append(b[0])
                    p_choices.append(X2[b[0]])
                if len(criticalX3) == 0 or np.all(criticalX3 == 0):
                    c = np.NaN
                    choices.append(c)
                    p_choices.append(0)
                else:
                    c = np.random.choice(X2_double_prime_ind, size = 1, p = criticalX3/criticalX3_sum)
                    choices.append(c[0])
                    p_choices.append(X2[c[0]])
                    
                chosen = np.random.choice(choices, p = p_choices/np.sum(p_choices))
                Kj = np.random.poisson(rxn_vector*tau) #generates number of transitions that can occur for each bacteria and receptor status
                Kj[:,0][X1_double_prime_ind] = 0
                Kj[:,1][X2_double_prime_ind] = 0
                Kj[:,2][X2_double_prime_ind] = 0
                if chosen == a:
                    Kj[:,0][a] = 1
                if chosen == b:
                    Kj[:,1][b] = 1
                if chosen == c:
                    Kj[:,2][c] = 1
            for c, rxn in enumerate(Kj):
                add_phages, subtract_phages = 0 , 0
                rx1 = rxn[0]*np.array([-1,+1,0])
                subtract_phages += rx1[0]

                rx2 = rxn[1]*np.array([+1,-1,0])
                add_phages += rx2[0]

                rx3 = rxn[2]*np.array([0,-1,+1])
                num_phages += (subtract_phages + add_phages)
                pop[c] += (rx1+rx2+rx3)
            t = tau
    return t, pop, num_phages
        



num_bacteria = int(3e+7)
#num_bacteria = 10
#num_phages = 5000000000
num_phages = int(num_bacteria*5e-4)
time = 0.0
max_steps = 2000
time_list = [0]*(max_steps+1)
print('initial number of phages' , num_phages)
k1,k2,k3 = 1.2e-11, 8.5e-4, 8e-4 


start = timeit.default_timer()
max_receptors = 250


pop = Generate_Population(num_bacteria,max_receptors)
stop = timeit.default_timer()
print('Time Gen Pop: ', stop - start) 

data = Generate_Data(pop)
all_data = np.zeros((max_steps,data.shape[0],data.shape[1]))
step = 0
all_data[0] = data
while (time<2*60*60):
    step +=1
    if step > max_steps:
        break
    print('step', step, 'time', time_list[step-1])
    start = timeit.default_timer()
    rxn_vector = Vectorization(num_bacteria,num_phages,k1,k2,k3,pop)
    print('rxn vector' , rxn_vector)
    stop = timeit.default_timer()
    print('Time Rxn Vector: ', stop - start) 

    t, pop, num_phages = Evolution(num_bacteria, num_phages, pop, rxn_vector)

    time += t
    time_list[step] = time
    data = update_data(pop, data)
    all_data[step] = data
    print(data)
    print('num phages' , num_phages)
    break

# plt.yscale('log')
# plt.xlabel('time (s)')
# plt.ylabel('numbers in pop')
# for i in range(data.shape[0]):
#     for j in range(data.shape[0]):
#         plt.plot(time_list[:step], all_data[:step,i,j])
# plt.show()


# In[ ]:


free_phages = [3e+7*5e-4]*step
plt.figure(figsize=[20,10])
for f in range(step):
    free_phages[f] = free_phages[f]-np.sum(all_data[f] )+np.sum(all_data[f][0][0])
plt.xlabel('time (s)', fontsize=20)
plt.ylabel('numbers in pop', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
for i in range(0,5):
    for j in range(0,5):
        if i == 0 and j == 0:
            pass;
        else:
            plt.plot(time_list[:step], all_data[:step,i,j],label='adhered{}, perm{}'.format(i,j))
plt.plot(time_list[:step], free_phages[:step], label='free phages')
plt.legend(bbox_to_anchor=(1.8, 1.05),fontsize=30,ncol=2)
plt.show()


# In[8]:


time_list = np.array(time_list)
from scipy.integrate import odeint
def phage_kinetics(pop,t,k1,k2,k3):
    
    B = pop[0]
    P = pop[1]
    BP = pop[2]
    BP_ = pop[3]
    #print(B,P,BP,BP_)

    dBPdt = k1*B*P-(k2+k3)*BP
    dPdt = k2*BP-k1*B*P
    dBdt = k2*BP-k1*B*P
    dBP_dt = k3*BP
    #print(dBdt,dPdt,dBPdt,dBP_dt)
    return [dBdt, dPdt, dBPdt, dBP_dt];

#time points
n = 6116
t = np.linspace(0,800,n) #800 seconds
 
B = np.zeros(n)
P = np.zeros(n)
BP = np.zeros(n)
BP_ = np.zeros(n)
#initial conditions:
num_bacteria = int(3e+7)
num_phages = num_bacteria*5e-4
pop0 = [num_bacteria,num_phages,0,0]

for i in range(1,n):
    tspan = [t[i-1],t[i]]
    pop = odeint(phage_kinetics,pop0,t,args=(k1,k2,k3))
    pop0 = pop[i]
    B[i] = pop0[0]
    P[i] = pop0[1]
    BP[i] = pop0[2]
    BP_[i] = pop0[3]


plt.figure(figsize=[20,10])
#time points
n = 6116
t = np.linspace(0,400,n) 

B = np.zeros(n)
P = np.zeros(n)
BP = np.zeros(n)
BP_ = np.zeros(n)
#initial conditions:
num_bacteria = int(3e+7)
num_phages = num_bacteria*5e-4
pop0 = [num_bacteria,num_phages,0,0]
k1,k2,k3 = 1.2e-11,8.5e-4,8e-4 
for i in range(1,n):
    tspan = [t[i-1],t[i]]
    pop = odeint(phage_kinetics,pop0,t,args=(k1,k2,k3))
    pop0 = pop[i]
    B[i] = pop0[0]
    P[i] = pop0[1]
    
    BP[i] = pop0[2]
    BP_[i] = pop0[3]
plt.title('Gillespie vs Deterministic', fontsize=40)
plt.plot(t[1:6115]/60,BP[1:6115]+BP_[1:6115],label = 'deterministic BP complexes',linestyle = '--',linewidth=4.0)
plt.plot(t[1:6115]/60,P[1:6115], label = 'deterministic free Phages',linestyle = '--',linewidth = 4.0)
for i in range(0,5):
    for j in range(0,5):
        if i == 0 and j == 0:
            pass;
        else:
            plt.plot(time_list[:step], all_data[:step,i,j],label='adhered{}, perm{}'.format(i,j))
plt.plot(time_list[:step], free_phages[:step], label='free phages')
plt.legend(bbox_to_anchor=(1.8, 1.05),fontsize=30,ncol=2)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('time (s)')
plt.ylabel('pop size')
plt.show()


# In[10]:


#data constants
import pandas as pd
import matplotlib.pyplot as plt
T = [1,3,4,10,15,20,24,30,35,37,40]
glucose_grown_k1 = [9.6e-13, 3.0e-12, 2.3e-12, 1.6e-11, 2.0e-11, 1.2e-11, 3.3e-11, 
                    1.5e-11, 2e-11,1.6e-11, 1.6e-11]
glucose_grown_k2 = [3.8e-4, 4e-3, 2.1e-3, 1.2e-3, 4.2e-3, 8.5e-4, 3e-3, 6.2e-4,
                   1.4e-3, 2.2e-3, 4.3e-4]
glucose_grown_k3 = [1.7e-3, 1.5e-3, 7.9e-4, 2.0e-4, 1.4e-3, 8.0e-4, 1.2e-3, 
                   5.7e-4, 1.2e-3, 1.5e-3, 5.1e-4]


# In[37]:


maltose_grown_k1 = [7.4e-13, 6.0e-12, 3.6e-12, 1.3e-11, 3.5e-11, 1.4e-11, 2.2e-11, 1.6e-11,
                   2.7e-11, 2.6e-11, 2.9e-11]
maltose_grown_k2 = [2.5e-3, 1.7e-3, 3.6e-3, 1.1e-3, 2.0e-3, 9.7e-4, 1.8e-3, 1.9e-3, 1.8e-3,
                   5.4e-3, 2.3e-3]
maltose_grown_k3 = [1e-3, 7.2e-4, 5.2e-4, 4.7e-4, 8.4e-4, 5.7e-4, 8.4e-4, 7.8e-4, 6e-4, 1.5e-3, 1.2e-3]


# In[40]:


plt.title('Glucose Grown Cells')
plt.yscale('log')
plt.xlabel('Temperature (\N{DEGREE SIGN} C)')
plt.ylabel('$cm^3s^{-1}$ / $s^{-1}$')
plt.plot(T, glucose_grown_k1, 'ro', label = 'k1 $cm^3/s$')
plt.plot(T, glucose_grown_k2, 'bo', label = 'k2 1/s')
plt.plot(T, glucose_grown_k3, 'go', label = 'k3 1/s')
plt.legend()
plt.show()

plt.title('Maltose Grown Cells')
plt.yscale('log')
plt.xlabel('Temperature (\N{DEGREE SIGN} C)')
plt.ylabel('$cm^3s^{-1}$ / $s^{-1}$')
plt.plot(T, maltose_grown_k1, 'ro', label = 'k1 $cm^3/s$')
plt.plot(T, maltose_grown_k2, 'bo', label = 'k2 1/s')
plt.plot(T, maltose_grown_k3, 'go', label = 'k3 1/s')
plt.legend()
plt.show()


# In[ ]:




