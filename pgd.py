#use proximal gradient descent to solve LASSO formulation, which is convex 
import math
import random
import numpy as np
from numpy import linalg as LA



#--------------------------------------------------------------------------------
#indicate function
#indicate the y vector
# if element in y positive, replace it with 1
# if negative, -1
# if zero, 0
def sign(y):
    cy = np.copy(y)
    length = len(cy)
    for i in range(length):
        if cy[i] > 0.0:
            cy[i] = 1.0
        elif cy[i] < 0.0:
            cy[i] = -1.0
        else:
            cy[i] = 0.0
    return cy

#compare a value with each element in a vector
# and replace the element with the max one, which is the value of origin 
#return a new vector
def vec_max(value, vector):
    for i in range(len(vector)):
        vector[i] = max(value, vector[i])
    return vector

#proximal function 
#lmd is the lambda value for convex closed function
# y is the [x_k - r_k* f'(x_k)]
#output prox_vec is p*1 dimensions
def prox_operation(lmd,step_length,y):
    new_y = sign(y) 
    print 'new_y',new_y
    prox_vec = new_y * vec_max(0, np.absolute(y) - step_length*lmd)   # NOTE: here is step_length * lmd 
    print prox_vec
    return prox_vec
#--------------------------------------------------------------------------------    
# gradient step
def gradient_descent(x_k, step_length,A,b):
    f_prime = np.dot(A.T, np.dot(A,x_k.T) - b) # derivertive of f_function 
    print 'f_prime',f_prime
    y = x_k - step_length * f_prime
    return y 

#--------------------------------------------------------------------------------
# primary function f(x) = (||Ax - b||^2) /2 
def f_func(A,x,b):
    f_value = math.pow(LA.norm(np.dot(A,x.T) - b),2) / 2
    return f_value

#model function
def m_func(A, x_k, b, step_length, x):
    part_1 = f_func(A,x_k,b)
    #part_1 = math.pow(LA.norm(np.dot(A,x_k.T) - b),2) / 2 
    print 'part_1',part_1
    part_2 = np.dot( (np.dot(A.T, np.dot(A,x_k.T) - b)).T, (x-x_k).T )
    print 'part_2',part_2
    part_3 = math.pow(LA.norm(x-x_k),2) / (2*step_length)
    print 'part_3',part_3
    m_value = part_1 + part_2 + part_3
    return m_value

#objective function
def obj_func(A,x,b,lmd):
    f_value = math.pow(LA.norm(np.dot(A,x.T) - b),2) / 2
    g_value = lmd * LA.norm(x,1)
    value = f_value + g_value
    return value
#--------------------------------------------------------------------------------

if __name__ == '__main__':
    n = 100
    p = 50
    s = 20   # nonzero size
    #when initialize a vector, had better to initialize it as row vector instead column vector  
    A = np.random.normal(0,1,(n,p))
    opt_x = np.array([0.0]*p)  # optimum x, used for generating b, and try to get optimum x using b and A

    #sample a list of index from 1 to 100 without duplicate
    #and create a sparse x vector with 20 non-zero elements
    random_index_list = random.sample(range(p), s) 
    for i in random_index_list:
        opt_x[i] = np.random.normal(0,10)
    print 'opt_x',opt_x
    e = np.random.normal(0,1,n)      
    print 'e',e
    b = np.dot(A,opt_x.T) + e.T
    print 'b',b


    x_k = np.array([0.0]*p)      
    lmd = math.sqrt(2*n*math.log(p))  #lambda in LOSSA objective function
    print 'lambda',lmd
    #step_length = 1/LA.norm(np.dot(A.T,A))   
    #step_length = lmd
    step_length = 10
    outfile = open('pgd.output','w')


    while(1):
        obj_value = obj_func(A,x_k,b,lmd)
        outfile.write(str(obj_value)+'\n')
        while(1):
            print 'x_k',x_k
            y = gradient_descent(x_k, step_length, A, b)
            print 'y',y
            x_k_plus_1 = prox_operation(lmd,step_length, y)
            print 'x_k_plus_1',x_k_plus_1
            f_value = f_func(A,x_k_plus_1,b) 
            print 'f_value',f_value
            m_value = m_func(A,x_k, b, step_length, x_k_plus_1)
            print 'm_value',m_value
            if f_value <= m_value:
                break
            step_length = step_length * 0.5
            print '********************************************'
        if f_func(A,x_k,b) <= f_func(A,x_k_plus_1,b):
            break
        else:
            x_k =  x_k_plus_1
    
    print opt_x
    print e
    print x_k_plus_1




