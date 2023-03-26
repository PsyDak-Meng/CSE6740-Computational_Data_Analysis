import numpy as np

def cluster(T, K, num_iters = 700, epsilon = 1e-12):
    """
    
	:param bow:
		bag-of-word matrix of (num_doc, V), where V is the vocabulary size
	:param K:
		number of topics
	:return:
		idx of size (num_doc), idx should be 1, 2, 3 or 4
	"""
    
    
    '''Initialization'''
    J=T.shape[1]
    I=T.shape[0]
    #try different initialization
    pi_c=np.full((K,1),1/K)
    posterior_Di_c=np.zeros((I,K))
    mu=np.zeros((J,K))
    for k in range(K):
        mu_prep=T+np.random.uniform(0,np.sum(np.sum(T))/I,size=(I,J))
        mu[:,k]=np.reshape(np.sum(mu_prep,axis=0)/np.sum(np.sum(mu_prep)),(J))
    mu_temp=np.copy(mu)
    for k in range(K):
        for i in range(I):
            for j in range(J):
                mu_temp[j,k]=mu[j,k]**T[i,j]
            posterior_Di_c[i,k]=np.prod(mu_temp[:,k])
    tau_ic=np.zeros((I,K))
    print('Initialization done.')
    
    print('EM Iteraiton...')
    for count in range(num_iters):
        
        '''E-Step'''    
        for k in range(K):
            for i in range(I):
                tau_ic[i,k]=pi_c[k]*posterior_Di_c[i,k]/np.dot(pi_c.T,posterior_Di_c[i,:].T)
        
        
        '''M-Step'''
        for k in range(K):
            lower_temp=0        
            for j in range(J):
                for i in range(I):
                    lower_temp+=(tau_ic[i,k]*T[i,j])
            for j in range(J):
                upper_temp=0
                for i in range(I):
                    upper_temp+=(tau_ic[i,k]*T[i,j])
                mu[j,k]=upper_temp/lower_temp
            pi_c[k]=np.sum(tau_ic[:,k])/I
            mu_temp=np.copy(mu)
            for i in range(I):
                for j in range(J):
                    mu_temp[j,k]=mu[j,k]**T[i,j]
                posterior_Di_c[i,k]=np.prod(mu_temp[:,k])

    print('EM Iteration done.')

    print('Computing final probability density estimaiton...')
    p_Di=np.zeros((I,K))
    for k in range(K):
        for i in range(I):
            p_Di[i,k]=pi_c[k]*posterior_Di_c[i,k]
    print('Done.')
    idx=np.zeros((I))
    for i in range(I):
        idx[i]=int(np.where(p_Di[i]==p_Di[i].max())[0])+1
	#raise NotImplementedError
    return idx

def cluster_extra(mat,N):
    
    return 
