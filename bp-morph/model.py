import numpy as np
from numpy import eye, zeros, exp, log, dot, outer
import itertools as it

class TagPair(object):
    """
    Tag pair model
    """

    def __init__(self, N):
        self.N = N
        self.F = eye(N)
        self.neg = zeros((N, N))
        


class RRBM(object):
    """
    Retricted Restricted Boltzmann Machine
    for tag prediction.
    
    Note: tag predictor

    """

    def __init__(self, N, train):
        self.N = N
        self.train = train
        self.W = np.random.rand(N, N)
        #self.W = eye(N)
        self.b = np.random.random(N)
        #self.b = zeros((N))

        
    def Z(self, W, b, x):
        """ Computes the log partition function Z(x) """
        Z = 1.0
        x_W  = dot(x, W)
        for i in xrange(self.N):
            Z *= (exp(x_W[i] + b[i]) + 1)
        return Z

    
    def enum_Z(self, W, b, x):
        """ Check the partition function Z(x) by enumeration """

        Z = 0
        x_W = dot(x, W)
        for y_lst in it.product([0, 1], repeat=self.N):
            y = np.asarray(y_lst)
            score = dot(x_W, y) + dot(b, y)
            Z += exp(score)
        return Z
            
    
    def f(self, W, b):
        """ Log-likelihood of the training data """
        ll = 0
        for x, y in self.train:
            score = dot(dot(x, W), y) + dot(b, y)
            Z = self.Z(W, b, x)
            ll += score - log(Z)
            
        return ll

    
    def g(self, W, b):
        """ Computes the gradient of the log-likelihood """
        W_g, b_g = zeros((self.N, self.N)), zeros((self.N))
        for x, y in self.train:
            #W_ += y
            tmp = exp(dot(x, W) + b)
            marg = tmp / (1 + tmp)
            W_g += outer(x, y - marg)
            b_g += y - marg
            
        return W_g, b_g
    

            
    def theta2vec(self, W, b):
        """ converts the parameters W and b to one long vector """
        return np.concatenate((np.asarray(W).reshape(-1), b))

    def vec2theta(self, vec):
        """ converts the vector to the parameter W and b """
        lst = np.split(vec, [self.N**2, len(vec)])
        W = lst[0].reshape(self.N, self.N)
        b = lst[1]
        return W, b
