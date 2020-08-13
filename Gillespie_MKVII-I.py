import numpy as np 
import scipy as sp 
import matplotlib.pyplot as plt 
import pandas as pd
import math 
import timeit 


def Genereate_Data(reactants):
    data = np.zeros((max_receptors,max_receptors),dtype=np.int32)
    #only count the number of phage-bacteria complexes
    species = np.array(list(reactants.keys()))
    counts = list(reactants.values())
    counts = np.reshape(counts,(len(counts),1))
    u = np.unique(species[:,1:3],axis = 0)
    #for i,sp in enumerate(species):
    sp_c = np.hstack((species[:,1:3],counts))
    
    for unq in u:
        ind = (sp_c[:,0:2]==unq)
        #print(ind)
        indexes = []
        for i in range(len(ind)):
            if np.all(ind[i]==True):
                indexes.append(i)
        c = np.sum(sp_c[indexes][:,-1])
        #print(c)
        x = unq[0]
        y = unq[1]
        data[x][y] = c
    print(data)
    return data

def Find_Reactants1(num_bacteria,min_receptors,max_receptors):
    denom = (max_receptors-min_receptors)
    groupings = num_bacteria/denom 
    reactants = {}
    for d in range(min_receptors,max_receptors):
        key = tuple([d,0,0])
        reactants[key] = int(groupings)
    return reactants  

def Find_Criticals(reactants):
    #critical deemed if species is at risk of going below 0
    #we risk negative based on receptor status
    #if c comes close to 0
    criticals = []
    reactants_keys = list(reactants.keys())
    for rk in reactants_keys:
        if reactants[rk]<nc:
            criticals.append(rk)
    #returns a list of keys corresponding to critical species
    #if num_phages<nc then every foward reaction is critical
    #Just engage SSA if this happens?
    return criticals

def Generate_Propensities(reactants,num_phages):
    #this function calculates the reaction scheme for each species
    #this is the aj step in Gillespie
    aj = {}
    total_prop = 0
    reactants_keys = list(reactants.keys())
    reactants_values = list(reactants.values())
    for r in reactants_keys:
        rx1 = reactants[r]*num_phages*k1
        #rx1 = num_bacteria*num_phages*k1
        rx2 = reactants[r]*r[1]*k2
        rx3 = reactants[r]*r[1]*k3
        total_prop += (rx1+rx2+rx3)
        aj[r] = [rx1,rx2,rx3]
    #returns propensities for each receptor status of each bacteria
    #shape {(x,y,z):[a1,a2,a3]}
    
    return aj,total_prop   

def Generate_Fj(aj,num_phages,reactants):
    #this is the partial derivative of each reaction in terms of reactants
    fj = {}
    reactants_keys = list(aj.keys())
    reactants_values = list(aj.values())
    for r in reactants_keys:
        rx1 = (reactants[r]+num_phages)*-k1
        #rx1 = (num_bacteria+num_phages)*-k1
        rx2 = -k2
        rx3 = -k3
        fj[r] = [rx1,rx2,rx3]
    return fj

def Generate_Uj_Sj(aj,fj):
    #First and Second order stats from reactions
    uj = {}
    sj = {}
    reactants_keys = list(aj.keys())
    reactants_values = list(aj.values())
    for r in reactants_keys:
        urx1 = aj[r][0]*fj[r][0]
        urx2 = aj[r][1]*fj[r][1]
        urx3 = aj[r][2]*fj[r][2]

        srx1 = aj[r][0]*fj[r][0]**2
        srx2 = aj[r][1]*fj[r][1]**2
        srx3 = aj[r][2]*fj[r][2]**2

        uj[r] = [urx1,urx2,urx3]
        sj[r] = [srx1,srx2,srx3]

    return uj,sj

def Find_Tau1(aj,uj,sj):
    total_prop = np.sum(list(aj.values()))
    ujs = np.array(list(uj.values()))
    sjs = np.array(list(sj.values()))

    left = total_prop/abs(ujs)
    right = np.power((total_prop*e),2)/sjs 
    l = np.nanmin(left)
    r = np.nanmin(right)
    tau1 = np.nanmin([l,r])
    return tau1 

def Find_Tau2(criticals,aj):
    critical_sum = 0
    for cr in criticals:
        critical_sum += np.sum(aj[cr])
    if critical_sum == 0:
        tau2 = np.nan
    else:
        tau2 = (1/critical_sum)*np.log(1/np.random.uniform())
    return tau2, critical_sum

def Kj_Noncritical_Poisson(aj,tau):
    aj_keys = list(aj.keys())
    aj_values = np.array(list(aj.values()))
    P_aj_t = np.random.poisson(aj_values*tau)
    Kj = dict(zip(aj_keys,P_aj_t))
    #returns dictionary saying species bacteria, and number of firings for rxns
    #{(x,y,z):[kj1,kj2,kj3]}
    return Kj

def Kj_Critical_Poisson(aj,tau,criticals,critical_sum):
    #only generate poisson for noncritical species
    #criticals is a list of keys corresponding to critical reactants
    #set all critical propensities to 0

    critical_probs = np.zeros((len(criticals),3))
    for i,cr in enumerate(criticals):
         #since the bacteria is a species itself
                                       #if its species count is low, all reactions on it are critical
        critical_probs[i] = aj[cr]
        aj[cr] = np.array([0,0,0])
    Kj = Kj_Noncritical_Poisson(aj,tau) #since it is essentially the same function just without criticals
    #then choose one critical reaction to occur based on point probabilities a_jcritical/critical_sum
    critical_probs = critical_probs/critical_sum #normalize on critical sums
    critical_probs_rows = np.sum(critical_probs,axis = 1) #choose along rows for heaviest bacteria
    critical_probs_rows = critical_probs_rows/np.sum(critical_probs_rows) #normalize for numpy
    #critical_probs = critical_probs/np.sum(critical_probs) #normalize for numpy
    choose_bact = np.random.choice(range(len(criticals)),p=critical_probs_rows) 
    #chooses an index corresponding to a key for a critical bacteria
    chosen_bact = criticals[choose_bact]
    critical_reaction = np.random.choice([0,1,2],
                         p = critical_probs[choose_bact]/np.sum(critical_probs[choose_bact]))
    if critical_reaction == 0:
        delta = [1,0,0]
    if critical_reaction == 1:
        delta = [0,1,0]
    if critical_reaction == 2:
        delta = [0,0,1]
    Kj[chosen_bact] = np.array(delta)
    return Kj

def Sub_Evolution1(reactants,num_phages,change_of_state,reaction_table,rk):
    x,y,z = rk[0],rk[1],rk[2]
    reactants_keys = list(reactants.keys())
    sorted_rxn = reaction_table[np.argsort(reaction_table[:,1])]
    #let m = max rxns, k' = second largest rxns, k'' = smallest rxns
    #first all three reactions co-occur
    #number of times this happens is: k'' 
    #second, once the smallest of co-occuring reactions is 
    #drained, its just 2 co-occuring rxns and this happens: k' times
    #finally, only one reaction can occur now, the remainder of the largest
    #this happens (m-k') times
    first = int(sorted_rxn[0][1]) #smallest
    second = int(sorted_rxn[1][1] - sorted_rxn[0][1]) #second largest
    third = int(sorted_rxn[2][1]-sorted_rxn[1][1]) #largest
    
    smallest_rxn_occuring = int(sorted_rxn[0][0])
    second_rxn_occuring = int(sorted_rxn[1][0])
    largest_rxn_occuring = int(sorted_rxn[2][0])
    if first!=0:
       # print('FIRST')
        for i in range(first):
            reactants_keys = list(reactants.keys())
            #all three reactions happen
            
            delta = np.array(change_of_state[smallest_rxn_occuring] + 
                    change_of_state[second_rxn_occuring] + 
                    change_of_state[largest_rxn_occuring])
            #print('=======DELTA========')
            #print(delta)
            delta_phages = delta[0]
            reactants[rk] -= 1
            num_phages += delta_phages
            #print('old reactant')
            #print(rk)
            new_reactant = tuple(np.array([x,y,z])+delta) 
            #print('new reactant')
            #print(new_reactant)
            if new_reactant in reactants_keys:
                reactants[new_reactant] += 1 
            else:
                reactants[new_reactant] = 1
    if second!=0:
       # print('SECOND')
        for i in range(second):
            reactants_keys = list(reactants.keys())
            #only two reactions happen
            delta = np.array(change_of_state[second_rxn_occuring]+
                    change_of_state[largest_rxn_occuring])
            delta_phages = delta[0] #change in phages is the same as the net of col 1 of change_of_state
            #print('=======DELTA========')
            #print(delta)
            num_phages += delta_phages
            reactants[rk] -= 1
            #print('old reactant')
            #print(rk)
            new_reactant = tuple(np.array([x,y,z])+delta)
            #print('new reactant')
            print(new_reactant)
            if new_reactant in reactants_keys:
                reactants[new_reactant] += 1 
            else:
                reactants[new_reactant] = 1
    if third!=0:
        #print('THIRD')
        #print(third)
        for i in range(third):
            reactants_keys = list(reactants.keys())
            #only one reaction happens
            delta = np.array(change_of_state[largest_rxn_occuring])
            #print('=======DELTA========')
            #print(delta)
            delta_phages = delta[0]
            #print(delta_phages)
            num_phages += delta_phages
            reactants[rk] -= 1
            #print('old reactant')
            #print(rk)
            new_reactant = tuple(np.array([x,y,z])+delta)
            #print('the new reactant')
            #print(new_reactant)
            if new_reactant in reactants_keys:
                reactants[new_reactant] += 1 
            else:
                reactants[new_reactant] = 1
    
    #print('NUMBER OF PHAGES: {}'.format(num_phages))
    return reactants,num_phages

def Sub_Evolution2(reactants,num_phages,change_of_state,reaction_table,rk):
    reactant_keys = list(reactants.keys())
    x,y,z = rk[0],rk[1],rk[2]
    sorted_rxn = reaction_table[np.argsort(reaction_table[:,1])]
    first = int(sorted_rxn[0][1]) #smallest
    second = int(sorted_rxn[1][1])
    third = int(sorted_rxn[2][1])
    
    smallest_rxn_occuring = int(sorted_rxn[0][0])
    second_rxn_occuring = int(sorted_rxn[1][0])
    largest_rxn_occuring = int(sorted_rxn[2][0])

    reactions = {smallest_rxn_occuring:first,
                second_rxn_occuring:second,
                largest_rxn_occuring:third
                 }
    #print(reactions)
    #[foward, reverse]
    max_firings = np.min([reactions[0],reactions[1]])
    for f in range(max_firings):
        reactant_keys = list(reactants.keys())
        delta = change_of_state[0]+change_of_state[1]
        #print(delta)
        delta_phages = delta[0]
        reactants[rk] -= 1
        num_phages += delta_phages
        new_reactant = tuple(np.array([x,y,z]+delta))
        #print(rk)
        #print(new_reactant)
        if new_reactant in reactant_keys:
            reactants[new_reactant] += 1
        else:
            reactants[new_reactant] = 1
        reactions[0] -= 1
        reactions[1] -= 1
    #[foward, irreverse]
    max_firings = np.min([reactions[0],reactions[2]])
    if max_firings == 0:
        max_firings = np.max([reactions[0],reactions[2]])
        if max_firings == reactions[0]:
            for f in range(max_firings):
                reactant_keys = list(reactants.keys())
                #only goes forward 
                delta = change_of_state[0]
               # print(delta)
                delta_phages = delta[0]
                reactants[rk] -= 1
                num_phages += delta_phages
                new_reactant = tuple(np.array([x,y,z]+delta))
                #print(rk)
                #print(new_reactant)
                if new_reactant in reactant_keys:
                    reactants[new_reactant] += 1
                else:
                    reactants[new_reactant] = 1
        if max_firings == reactions[2]:
            for f in range(max_firings):
                reactant_keys = list(reactants.keys())
                delta = change_of_state[2]
                #print(delta)
                delta_phages = delta[2]
                reactants[rk] -= 1
                num_phages += delta_phages
                new_reactant = tuple(np.array([x,y,z]+delta))
                #print(rk)
                #print(new_reactant)
                if new_reactant in reactant_keys:
                    reactants[new_reactant] += 1
                else:
                    reactants[new_reactant] = 1
                reactions[0] -= 1
                reactions[2] -= 1
    else:
        for f in range(max_firings):
            reactant_keys = list(reactants.keys())
            delta = change_of_state[0]+change_of_state[2]
            #print(delta)
            delta_phages = delta[0]
            reactants[rk] -= 1
            num_phages += delta_phages
            new_reactant = tuple(np.array([x,y,z]+delta))
            #print(rk)
            #print(new_reactant)
            if new_reactant in reactant_keys:
                reactants[new_reactant] += 1
            else:
                reactants[new_reactant] = 1
            reactions[0] -= 1
            reactions[2] -= 1            
    left_overs = np.max([reactions[0],reactions[2]])
    if left_overs == reactions[0]:
        for f in range(left_overs):
            reactant_keys = list(reactants.keys())
            delta = change_of_state[0]
            #print(delta)
            delta_phages = delta[0]
            reactants[rk] -= 1
            num_phages += delta_phages
            new_reactant = tuple(np.array([x,y,z]+delta))
            #print(rk)
            #print(new_reactant)
            if new_reactant in reactant_keys:
                reactants[new_reactant] += 1
            else:
                reactants[new_reactant] = 1
            reactions[0] -= 1    
    if left_overs == reactions[2]:       
        for f in range(left_overs):
            reactant_keys = list(reactants.keys())
            delta = change_of_state[2]
            #print(delta)
            delta_phages = delta[2]
            reactants[rk] -= 1
            num_phages += delta_phages
            new_reactant = tuple(np.array([x,y,z]+delta))
            #print(rk)
            #print(new_reactant)
            if new_reactant in reactant_keys:
                reactants[new_reactant] += 1
            else:
                reactants[new_reactant] = 1
            reactions[2] -= 1    
    return reactants,num_phages

def Evolution(reactants,num_phages,Kj):
    dummy_keys = list(reactants.keys())
    delta_reactants = {}
    #print('NUMBER OF PHAGES: {}'.format(num_phages))
    reactant_keys = list(reactants.keys())
    for rk in dummy_keys:
        
        x,y,z = rk[0],rk[1],rk[2]
        k = Kj[rk]
        #binding = 0 B+P --> BP
        #dissociate = 1  BP --> B+P
        #irreversible = 2 BP--> BP*
        change_of_state = {0:np.array([-1,+1,0]),
                           1:np.array([+1,-1,0]),
                           2:np.array([0,-1,+1])}
        reaction_table = np.zeros((3,2))
        reaction_table[0] = [0,k[0]]
        reaction_table[1] = [1,k[1]]
        reaction_table[2] = [2,k[2]]
        #Sub_Evolution1 and Sub_Evolution2 are not well coded at all
        #I didn't want to think anymore, I am certain there is a much better way
        # for now, they are hardcoded.
        if y>1:
            reactants,num_phages = Sub_Evolution1(reactants,num_phages,change_of_state,reaction_table,rk)
        else:
            #then the only 2 reactions that can co-occur:
            #[foward, disocciate]
            #[foward, irreversible]
            reactants,num_phages = Sub_Evolution2(reactants,num_phages,change_of_state,reaction_table,rk)

    return reactants,num_phages 

def Direct_SSA(reactants,num_phages):
    tau=0
    for iteration in range(100):
        species = list(reactants.keys())
        aj,total_prop = Generate_Propensities(reactants,num_phages)
        aj_keys = np.array(list(aj.keys()))
        aj_values = np.array(list(aj.values()))
        taus = (1/aj_values)*np.log(1/np.random.uniform(size=(aj_values.shape[0],aj_values.shape[1])))
        #find the smallest in rows and columns
        #t = (1/total_prop)*np.log(1/np.random.uniform())
        t = np.nanmin(taus)
        t_where = np.where(taus==t)
        trow = t_where[0][0]
        tcol = t_where[1][0]
        #print(tcol)
        propensities = np.array(list(aj.values()))
        chosen_bacteria = propensities[trow]
        chosen_rxn = tcol
        #prob_propensities = propensities/total_prop 
        #prob_propensities = prob_propensities/(np.sum(prob_propensities))
        #find heaviest bacteria
        #prob_propensities = np.sum(prob_propensities,axis=1) #sum the rows
        #prob_propensities = prob_propensities/np.sum(prob_propensities) #normalize for numpy
        #choose heaviest bacteria probabilistiaclly
        #heaviest_bact = np.random.choice(range(len(species)),p=prob_propensities)
        #chosen_bacteria = propensities[heaviest_bact]
        #chosen_rxn = np.random.choice([0,1,2],
        #                                p = chosen_bacteria/np.sum(chosen_bacteria))
        key = np.array(species[trow])
        #print(num_phages)
        if chosen_rxn == 0:
            #reversible binding
            #print('rx1')
            v = np.array([-1,+1,0])
            num_phages -= 1
        if chosen_rxn == 1:
            #print('rx2')
            v = np.array([+1,-1,0])
            num_phages += 1
        if chosen_rxn == 2:
            #print('rx3')
            v = np.array([0,-1,+1])
        #print(num_phages)

        new_reactant = key+v 
        #print(key)
        #print(new_reactant)
        #print(new_reactant)
        #print(species)
        new_reactant = tuple(new_reactant)
        if new_reactant in species:
            reactants[new_reactant] += 1
        else:
            reactants[new_reactant] = 1
        reactants[tuple(key)]-=1
        
        tau+=t
      

    return reactants,num_phages,tau

def Run_Algorithm(num_bacteria,num_phages):
    all_data = np.zeros((max_steps,max_receptors,max_receptors),dtype=np.int32)
    print(num_phages)
    #reactants = Find_Reactants(pop)
    reactants = Find_Reactants1(num_bacteria,min_receptors,max_receptors)
    stop = 110*60
    step = 0
    t=0
    time = np.zeros(max_steps)
    phages = np.zeros((max_steps))
    while(step<max_steps and t<stop):
        num_phages0 = num_phages
        criticals = Find_Criticals(reactants)
        print('CRITICALS:')
        print(criticals)
        aj,total_prop = Generate_Propensities(reactants,num_phages)
        fj = Generate_Fj(aj,num_phages,reactants)
        uj,sj = Generate_Uj_Sj(aj,fj)
        tau1 = Find_Tau1(aj,uj,sj)
        tau2,critical_sum = Find_Tau2(criticals,aj)
        tau = np.nanmin([tau1,tau2])
        #print('TAU1 vs TAU2: {} |vs| {}'.format(tau1,tau2))
        #print('===========================REACTANTS================================')
        #print(reactants)
        #print('========================PROPENSITIES======================')
        #print(aj)
        if tau==tau1:   
            if (tau<(.1*(1/total_prop))):

                #engage SSA
                #print('ENGAGE SSA') #have not written ssa yet
                reactants,num_phages,tau = Direct_SSA(reactants,num_phages)
            else:
                #print('NON CRITICAL POISSON ENGAGED')
                #Kj = Kj_Noncritical_Poisson(aj,tau)
                #print('========================POISSON================================')
                #print(Kj)
                #reactants,num_phages = Evolution(reactants,num_phages,Kj)
                reactants,num_phages,tau = Direct_SSA(reactants,num_phages)
        if tau == tau2:
            #print('CRITICAL POISSON ENGAGED')
            #Kj = Kj_Critical_Poisson(aj,tau,criticals,critical_sum)
            #reactants,num_phages = Evolution(reactants,num_phages,Kj)
            reactants,num_phages,tau = Direct_SSA(reactants,num_phages)
            #print('========================POISSON================================')
            #print(Kj)
        
        print('======================================================================')
        print('ITERATION: {}'.format(iter))
        print('STEPl {}'.format(step))
        print('TAU: {} (s)'.format(tau))
        print('VIRUSES LEFT: {}'.format(num_phages))
        
        data = Genereate_Data(reactants)
        all_data[step] = data
        time[step] = t
        phages[step] = num_phages
        t+=tau 
        print('TIME: {} (min)'.format(t/60))
        print('RATE OF VIRUS LOSS: {}/s'.format(num_phages0-num_phages))
        #if num_phages0-num_phages<0:
        #    break
        step+=1
        if num_phages <0:
            break
        #if step>5:
        #    break
    return all_data,time,step,phages

nc = 200
k1,k2,k3 = 2.0e-11,2.2e-3,1.5e-3
e = 0.0003

max_receptors = 300
min_receptors = 250
max_steps = 20000
mega_data = np.zeros((max_steps,max_receptors,max_receptors),dtype=np.int32)
mega_phages = np.zeros((max_steps))
mega_time = np.zeros((max_steps))
iterations = 1
plt.figure(figsize = [40,20])
plt.xticks(fontsize=60)
plt.yticks(fontsize=60)
for iter in range(iterations):
    num_bacteria = int(3e8)
    num_phages = int(num_bacteria*5e-4)
    all_data,time,step,phages = Run_Algorithm(num_bacteria,num_phages)
    mega_data += all_data
    mega_phages += phages
    mega_time += time
    plt.xlabel('time (min)', fontsize=60)
    plt.ylabel('Pop Size/mL', fontsize=60)
    sum_data = np.zeros((len(time[:step])))
    for i in range(0,3):
        for j in range(0,3):
            if i == 0 and j == 0:
                pass
            else:
                sum_data += all_data[:step,i,j]
                plt.plot(time[:step]/60, all_data[:step,i,j],label='stochastic adhered {}, perm {}'.format(i,j),linestyle=':',linewidth = 3.0)
    
    plt.plot(time[:step]/60, sum_data[:step],label='Stochastic Total PB Complexes',linestyle=':',linewidth =3.0,color='red')
    plt.plot(time[:step]/60, phages[:step],label = 'Free Phages', linestyle = ':',linewidth = 3.0)

plt.legend(bbox_to_anchor=(.5,.5),fontsize=30,ncol=2)
plt.savefig('Gillespie_mkVII_multiple.png')
mega_data = mega_data/iterations
mega_phages = mega_phages/iterations
mega_time = mega_time/iterations

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
    return [dBdt, dPdt, dBPdt, dBP_dt]

#time points
n = 6116
t = np.linspace(0,110,n) #120 minutes divided by 6116 intervals
 
B = np.zeros(n)
P = np.zeros(n)
BP = np.zeros(n)
BP_ = np.zeros(n)
#initial conditions:
num_phages = num_bacteria*5e-4
#num_phages = num_bacteria*2
pop0 = [num_bacteria,num_phages,0,0]
k1,k2,k3 = 2.0e-11,2.2e-3,1.5e-3
for i in range(1,n):
    tspan = [t[i-1],t[i]]
    pop = odeint(phage_kinetics,pop0,t,args=(k1,k2,k3))
    pop0 = pop[i]
    B[i] = pop0[0]
    P[i] = pop0[1]
    BP[i] = pop0[2]
    BP_[i] = pop0[3]

plt.figure(figsize=[40,20])
plt.title('Gillespie vs Deterministic', fontsize=100)
plt.plot(t[1:6115],BP[1:6115]+BP_[1:6115],label = 'Deterministic Total BP Complexes',linestyle = '--',linewidth=8.0, color = 'orange',alpha=0.75)
plt.plot(t[1:6115],BP_[1:6115],label = 'Deterministic BP* Complexes', linestyle = '-',linewidth = 6, color = 'green',alpha=0.75)
plt.plot(t[1:6115],BP[1:6115],label = 'Deterministic BP Complexes', linestyle = '-',linewidth = 6, color = 'blue',alpha = 0.75)
plt.plot(t[1:6115],P[1:6115], label = 'Deterministic Free Phages',linestyle = '--',linewidth = 8.0, color = 'red', alpha =0.75)

plt.xticks(fontsize=60)
plt.yticks(fontsize=60)

plt.xlabel('Time (min)', fontsize=60)
plt.ylabel('Pop Size/mL', fontsize=60)
sum_data = np.zeros((len(time[:step])))
for i in range(0,3):
    for j in range(0,3):
        if i == 0 and j == 0:
            pass
        else:
            sum_data += mega_data[:step,i,j]
            plt.plot(mega_time[:step]/60, mega_data[:step,i,j],label='Stochastic Adhered {}, Perm {}'.format(i,j),linestyle=':',linewidth = 8.0,alpha=0.75,color='green')

plt.plot(mega_time[:step]/60, sum_data[:step],label='Stochastic Total PB Complexes',linestyle=':',linewidth = 8.0,color='orange',alpha=0.75)
plt.plot(mega_time[:step]/60, mega_phages[:step],label = 'Stochastic Free Phages', linestyle = ':',linewidth = 6.0, color = 'red',alpha=0.75)

plt.legend(bbox_to_anchor=(.5,.5),fontsize=30,ncol=2)
plt.savefig('Gillespie_mkVII_averages.png')
#plt.show()
    
