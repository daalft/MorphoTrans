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

    def __init__(self, N, embedding_size, character_size, train, C = 0.01):
        self.N = N
        self.train = train
        self.C = C  # regularization coefficient

        # embeddings test to see if it improves
        self.embedding_size = embedding_size
        self.character_size = character_size
        
        self.reset()

        

    def reset(self, random=False):
        """ reset the parameters to the initial values (just copy the tag) """
        
        if random:
            self.W = np.random.rand(self.N, self.N)
            self.E = np.random.rand(self.embedding_size, self.N)
            self.S = np.random.rand(self.character_size, self.N)
            self.b = np.random.random(self.N)

        else:
            #self.W = eye(self.N)
            self.W = zeros((self.N, self.N))
            self.E = zeros((self.embedding_size, self.N))
            self.S = zeros((self.character_size, self.N))
            self.b = zeros((self.N))
        
        
    def Z(self, W, E, S, b, x, e, c):
        """ Computes the log partition function Z(x) """
        
        Z = 1.0
        x_W  = dot(x, W)
        e_E = dot(e, E)
        c_S = dot(c, S)
        for i in xrange(self.N):
            Z *= (exp(x_W[i] + e_E[i] + c_S[i] + b[i]) + 1)
        return Z

    
    def enum_Z(self, W, E, S, b, x, e, c):
        """ Check the partition function Z(x) by enumeration """

        Z = 0
        x_W = dot(x, W)
        e_E = dot(e, E)
        c_S = dot(c, S)
        for y_lst in it.product([0, 1], repeat=self.N):
            y = np.asarray(y_lst)
            score = dot(x_W, y) + dot(e_E, y) + dot(c_S, y) + dot(b, y)
            Z += exp(score)
        return Z
            
    
    def f(self, W, E, S, b):
        """ Log-likelihood of the training data """
        
        ll = 0
        for x, y, e, c in self.train:
            score = dot(dot(x, W), y) + dot(dot(e, E), y) + dot(dot(c, S), y) + dot(b, y)
            Z = self.Z(W, E, S, b, x, e, c)
            ll -= (score - log(Z))
            
        return ll

    
    def g(self, W, E, S, b):
        """ Computes the gradient of the log-likelihood """
        
        W_g, E_g, S_g, b_g = zeros((self.N, self.N)), zeros((self.embedding_size, self.N)), zeros((self.character_size, self.N)), zeros((self.N))
        for x, y, e, c in self.train:
            tmp = exp(dot(x, W) + dot(e, E) + dot(c, S) + b)
            marg = tmp / (1 + tmp)
            W_g -= outer(x, y - marg)
            b_g -= (y - marg)
            #E_g -= outer(e, y - marg)
            S_g -= outer(c, y - marg)
            
        return W_g, E_g, S_g, b_g

    def marg(self, x, e, c):
        """ returns the marginal distribution  y given an x using the parameters of the model """
        tmp = exp(dot(x, self.W) + dot(e, self.E) + dot(c, self.S) + self.b)
        marg = tmp / (1 + tmp)
        return marg


    def predict(self, x, e, c):
        """ predicts a y given an x using the parameters of the model """
        return np.round(self.marg(x, e, c))


    def eval(self, heldout):
        """ evaluate under two metrics: 1-best error rate and hamming """

        mistakes = dd(int)
        err, ham  = 0, 0
        N = float(len(heldout))

        for x, y, e, c in heldout:
            y_pred = self.predict(x, e, c)
            err += 0.0 if (y == y_pred).all() else 1
            ham += hamming(y, y_pred) * len(y)

            for i, (y1, y2) in enumerate(zip(y, y_pred)):
                if y1 != y2:
                    mistakes[i] += 1
                    
        return err / N, ham / N, dict(mistakes)
        

    def learn(self, maxiter=1000, disp=0):
        """ trains the model using batch L-BFGS """
        reg = self.theta2vec(eye(self.N), zeros((self.embedding_size, self.N)), zeros((self.character_size, self.N)), zeros((self.N)))

        
        def ff(vec):
            W, E, S, b = self.vec2theta(vec)
            return self.f(W, E, S, b) + self.C / 2.0 * np.dot(vec - reg, vec - reg)

        def gg(vec):
            W, E, S, b = self.vec2theta(vec)
            gW, gE, gS, gb = self.g(W, E, S, b)
            return self.theta2vec(gW, gE, gS, gb) + self.C * (vec - reg)
        
        v = self.theta2vec(self.W, self.E, self.S, self.b)
        v, _, _ = lbfgs(ff, v, fprime=gg, maxiter=maxiter, disp=disp)
        self.W, self.E, self.S, self.b = self.vec2theta(v)

        
    def theta2vec(self, W, E, S, b):
        """ converts the parameters W and b to one long vector """
        
        return np.concatenate((np.asarray(W).reshape(-1), np.asarray(E).reshape(-1), np.asarray(S).reshape(-1), b))

    
    def vec2theta(self, vec):
        """ converts the vector to the parameter W and b """

        lst = np.split(vec, [self.N**2, self.N**2+self.embedding_size*self.N, self.N*(self.N+self.embedding_size+self.character_size)])
        W = lst[0].reshape(self.N, self.N)
        E = lst[1].reshape(self.embedding_size, self.N)
        S = lst[2].reshape(self.character_size, self.N)
        b = lst[3]
        return W, E, S, b

    

class StringFeatures(object):
    """ String features """
    
    def __init__(self, prefix_length=0, suffix_length=4):
        self.prefix_length, self.suffix_length = prefix_length, suffix_length
        self.attributes = Alphabet()
        self.word2attributes = {}
        self.words = Alphabet()

        
    def get_attributes(self, word, extract=False):
        """ extract the features """
        
        lst = []
        for i in xrange(1, self.prefix_length+1):
            if i > len(word):
                break
            prefix = word[:i]
            name = "PREFIX: "+prefix
            if extract:
                self.attributes.add(name)
            if name in self.attributes:
                lst.append(self.attributes[name])
            
        for i in xrange(1, self.suffix_length+1):
            if i < 0:
                break
            suffix = word[-i:]
            name = "SUFFIX: "+suffix
            if extract:
                self.attributes.add(name)
            if name in self.attributes:
                lst.append(self.attributes[name])

        return lst
                

    def store(self, word):
        """ store the features """

        self.words.add(word)
        i = self.words[word] 
        self.word2attributes[i] = self.get_attributes(word, True)

        
    def __len__(self):
        return len(self.attributes)

    
    def __getitem__(self, word):
        if word in self.words:
            i = self.words[word]
            return self.word2attributes[i]
        # don't extract
        return self.get_attributes(word, False)
        
        
class SurfaceForm(object):
    """ Surface Form Classifier """

    def __init__(self, N, train, C=0.01):
        self.N = N
        self.train = train
        self.C = C
        self.features = StringFeatures(N)

        for word, vec in train:
            self.features.store(word)
            
        self.theta = zeros((len(self.features)))
            
    def f(self, theta):
        """ Log-likelihood of the training data """

        for word, vec in train:
            pass

    def g(self, theta):
        pass
