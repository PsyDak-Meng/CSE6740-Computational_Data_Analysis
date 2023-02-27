import numpy as np

def cluster(T, K, num_iters = 1000, epsilon = 1e-12):
    """

	:param bow:
		bag-of-word matrix of (num_doc, V), where V is the vocabulary size
	:param K:
		number of topics
	:return:
		idx of size (num_doc), idx should be 1, 2, 3 or 4
	"""
    
    
    '''Initialization'''
    #T=np.ones((400,100))
    J=T.shape[1]
    I=T.shape[0]
    #try different initialization
    pi_c=np.full((I,K),1/J/K)
    posterior_Di_c=np.zeros((I,K))
    mu=np.resize(np.divide(T,np.sum(T,axis=1).reshape(I,1)),(I,J,K))
    for k in range(K):
        for i in range(I):
            for j in range(J):
                mu[i,j,k]=mu[i,j,k]**T[i,j]
            posterior_Di_c[i,k]=np.prod(mu[i,:,k])
    tau_ic=np.ones((I,K))
    
    '''E-Step'''
    for k in range(K):
        for i in range(I):
            tau_ic[i,k]=pi_c[i,k]*posterior_Di_c[i,k]/np.dot(pi_c,posterior_Di_c.T)
    
    '''M-Step'''
    for count in range(num_iters):
        for i in range(I):
            for j in range(J):
                posterior_prep=np.dot(T[i],tau_icp[i])/
   
	#raise NotImplementedError

	return idx
