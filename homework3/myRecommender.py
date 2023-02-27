import numpy as np


def my_recommender(rate_mat, lr, with_reg):
    """

    :param rate_mat:
    :param lr:
    :param with_reg:
        boolean flag, set true for using regularization and false otherwise
    :return:
    """

    # TODO pick hyperparams
    max_iter = 1000
    learning_rate = 0.0000001
    reg_coef = 0.01
    n_user, n_item = rate_mat.shape[0], rate_mat.shape[1]
    #print(n_user,n_item)
    U = np.random.rand(n_user, lr) / lr
    V = np.random.rand(n_item, lr) / lr
    # TODO implement your code here
    def reg(with_reg):
        try: 
            if with_reg:
                return reg_coef
            else:
                return 0
        except:
            print('This parameter has to be True or False.')
    '''def MSE(M,U,V):
        loss=M-np.dot(U,V.T)
        loss=np.sum(loss)
        return loss'''
    
    for i in range(max_iter):
        V_sum_lr=np.sum(V,axis=1)
        #print('Vk: ', V_sum_lr)
        U_sum_lr=np.sum(U,axis=1)
        #print('Uk: ', U_sum_lr)
        U_temp=np.copy(U)
        #print('U: ', U_temp)
        V_temp=np.copy(V)
        #print('V: ', V_temp)
        U_grad=np.sum(-2*np.dot(rate_mat,V_sum_lr)
                      +np.sum(2*np.dot(np.dot(U_temp,V_temp.T),V_temp),axis=1))+reg(with_reg)*2*np.sum(U_temp)
        #print('U_grad:',U_grad)
        U=U_temp-learning_rate*U_grad
        V_grad=np.sum(-2*np.dot(rate_mat.T,U_sum_lr)
                      +np.sum(2*np.dot(np.dot(V_temp,U_temp.T),U_temp),axis=1))+reg(with_reg)*2*np.sum(V_temp)
        V=V_temp-learning_rate*V_grad

        if abs(U_grad)<= 0.0001 or abs(V_grad)<=0.0001:
            break
    print('U_grad:',U_grad)
    print('V_grad:',V_grad)
    return U,V