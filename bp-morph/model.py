import numpy as np
from numpy import eye, zeros, exp, log, dot, outer
from scipy.spatial.distance import hamming
import itertools as it
from scipy.optimize import fmin_l_bfgs_b as lbfgs
from collections import defaultdict as dd
from arsenal.alphabet import Alphabet

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

    def __init__(self, N, train, C = 0.01):
        self.N = N
        self.train = train
        self.C = C  # regularization coefficient
        self.reset()


    def reset(self, random=False):
        """ reset the parameters to the initial values (just copy the tag) """
        
        if random:
            self.W = np.random.rand(self.N, self.N)
            self.b = np.random.random(self.N)
        else:
            self.W = eye(self.N)
            self.b = zeros((self.N))
        
        
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
            ll -= (score - log(Z))
            
        return ll

    
    def g(self, W, b):
        """ Computes the gradient of the log-likelihood """
        
        W_g, b_g = zeros((self.N, self.N)), zeros((self.N))
        for x, y in self.train:
            tmp = exp(dot(x, W) + b)
            marg = tmp / (1 + tmp)
            W_g -= outer(x, y - marg)
            b_g -= (y - marg)
            
        return W_g, b_g
    

    def predict(self, x):
        """ predicts a y given an x using the parameters of the model """
        
        tmp = exp(dot(x, self.W) + self.b)
        marg = tmp / (1 + tmp)
        return np.round(marg)


    def eval(self, heldout):
        """ evaluate under two metrics: 1-best error rate and hamming """

        mistakes = dd(int)
        err, ham  = 0, 0
        N = float(len(heldout))

        for x, y in heldout:
            y_pred = self.predict(x)
            err += 0.0 if (y == y_pred).all() else 1
            ham += hamming(y, y_pred) * len(y)

            for i, (y1, y2) in enumerate(zip(y, y_pred)):
                if y1 != y2:
                    mistakes[i] += 1
                    
        return err / N, ham / N, dict(mistakes)
        

    def learn(self, disp=0):
        """ trains the model using batch L-BFGS """
        reg = self.theta2vec(eye(self.N), zeros((self.N)))

        
        def ff(vec):
            W, b = self.vec2theta(vec)
            return self.f(W, b) + self.C / 2.0 * np.dot(vec - reg, vec - reg)

        def gg(vec):
            W, b = self.vec2theta(vec)
            gW, gb = self.g(W, b)
            return self.theta2vec(gW, gb) + self.C * (vec - reg)
        
        v = self.theta2vec(self.W, self.b)
        v, _, _ = lbfgs(ff, v, fprime=gg, disp=disp)
        self.W, self.b = self.vec2theta(v)

        
    def theta2vec(self, W, b):
        """ converts the parameters W and b to one long vector """
        
        return np.concatenate((np.asarray(W).reshape(-1), b))

    
    def vec2theta(self, vec):
        """ converts the vector to the parameter W and b """
        
        lst = np.split(vec, [self.N**2, len(vec)])
        W = lst[0].reshape(self.N, self.N)
        b = lst[1]
        return W, b

    

class StringFeatures(object):
    """ String features """
    
    def __init__(self, N, prefix_length=4, suffix_length=4):
        self.N = N
        self.prefix_length, self.suffix_length = prefix_length, suffix_length
        self.features = Alphabet()
        self.word2features = {}
        
    def features(self, word, extract=False):
        """ extract the features """
        
        lst = []
        for i in xrange(1, self.prefix_length+1):
            if i > len(word):
                break
            prefix = word[:i]
            name = "PREFIX: "+prefix
            if extract:
                self.features.add(name)
            lst.append(self.features[name])
            
        for i in xrange(1, self.suffix_length+1):
            if i < 0:
                break
            suffix = word[-i:]
            name = "SUFFIX: "+suffix
            if extract:
                self.features.add(name)
            lst.append(self.features[name])
        return add

    def store(self, word):
        """ store the features """
        self.word2features[word] = self.features(word, True)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, word):
        if word in self.features:
            return self.features[word]
        else:
            return self.features(word)
        
        
class SurfaceForm(object):
    """ Surface Form Classifier """

    def __init__(self, N, train, C=0.01):
        self.N = N
        self.train = train
        self.C = C

            
    
