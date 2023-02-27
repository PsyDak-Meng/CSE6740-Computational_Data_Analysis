import matplotlib.pyplot as plt
import numpy as np


def algo(q, Y):
    # init
    p = 0.0
    fig, ax = plt.subplots()

    # TODO implement your algorithm and return the (i) prob p and (ii) a matplotlib Figure object for the plot
    
    print(Y)
    
    #Hidden state transition prob.
    a=[[0.8,0.2],[0.2,0.8]] #g-g,b; b-g,b
    #Start state prob. 
    pi=[0.2,0.8] # good,bad
    #Emission prob.
    p_gb=[[q,1-q],[1-q,1]] # good:+1,-1; bad:+1,-1
    
    T=39
    emit={1:0,-1:1}
    
    def forward(t):
        Y_t=Y[:T]
        alpha=np.empty([2,t])
        p_X=np.empty([1,t])
        
        #Initialization
        alpha[0,0]=pi[0]*p_gb[0][emit[Y_t[0,0]]]
        alpha[1,0]=pi[1]*p_gb[1][emit[Y_t[0,0]]]
        
        for i in range(1,t):
            last_sum=0
            for state in range(len(p_gb)):#k
                    for state_1 in range(len(p_gb)):#i
                        last_sum+=alpha[state_1,i-1]*a[state_1][state]
                    alpha[state,i]=p_gb[state][emit[Y_t[i,0]]]*last_sum
                    last_sum=0
            p_X=np.sum(alpha[:,T-1])
            
        return p_X, alpha
    
    def backward(t):
        beta=np.empty([2,t])
        Y_t=Y[:39]
        
        #Initialization
        beta[0,t-1]=1
        beta[1,t-1]=1
        
        for i in range(t-2,-1,-1):
            last_sum=0
            for state in range(len(p_gb)):#k
                    for state_1 in range(len(p_gb)):#i
                        last_sum+=a[state][state_1]*p_gb[state_1][emit[Y_t[i+1,0]]]*beta[state_1,i+1]
                    beta[state,i]=last_sum
                    last_sum=0
        return beta
    

    beta=backward(T)
    p_X,alpha=forward(T)
    p_all=np.multiply(alpha,beta)/p_X
    p=p_all[0][T-1]
    
    ax.plot(range(1,40,1),p_all[1])
    #print('p_X:',p_X)
    #print('alphha:',alpha)
    #print('beta:',beta)
    print(p_all[0]+p_all[1])
    
    return p, fig