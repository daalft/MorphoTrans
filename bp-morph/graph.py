""" Factor Graph for morphology """
import numpy as np
from numpy import zeros, ones, eye

class Edge(object):
    """ Edge """

    def __init__(self, f, v):
        # the factor
        self.f  = f
        self.m_f = .5 * ones(self.f.N)
        #self.f.edges.append(self)
        # index of the edge in the variable's
        # edge list
        #self.fi = len(self.f.edges)-1
        
        # the variable
        self.v = v
        self.m_v = .5 * ones(self.v.N)
        #self.v.edges.append(self)
        # index of the edge in the factor's
        # edge list
        #self.vi = len(self.v.edges)-1

        
    def p2f(self):
        """ Passes the message to the factor """

        stale = self.m_f
        self.m_f = self.v.b / self.m_v
        self.f.b = self.f.b / stale * self.m_f

        
        
    def p2v(self):
        """ Passes the message to the variable """

        # TODO
        stale = self.m_v
        self.m_v = self.f.b / self.m_f
        
        # update the belief by dividing out the
        # stale message and multiplying (point-wise)
        # in the new message
        self.v.b = self.v.b / stale * self.m_v

    def __unicode__(self):
        return unicode(self.v)+" => "+unicode(self.f)
    
        
class Variable(object):
    """ Variable in the factor graph """
    
    def __init__(self, lang, word):
        self.lang, self.word = lang, word
        self.edges = []
        # the belief
        self.b = None

    def __unicode__(self):
        return unicode(self.word)+" ("+unicode(self.lang)+")"

    def __repr__(self):
        return unicode(self)

    
class Factor(object):
    """ Factor in the factor graph """

    def __init__(self, N):
        self.b = None
        
    
class UnaryFactor(Factor):
    """ Unary Factor """
    
    def __init__(self, N):
        pass

    
class BinaryFactor(Factor):
    """ Binary Factor """
    
    def __init__(self, source_lang, source, target_lang, target, N):
        super(BinaryFactor, self).__init__(N)
        self.N = N
        self.F = eye(N)
        self.source_lang, self.target_lang = source_lang, target_lang
        self.source, self.target = source, target

    def __unicode__(self):
        return unicode(self.source)+" ("+unicode(self.source_lang)+") -- "+unicode(self.target)+" ("+unicode(self.target_lang)+")"

    def __repr__(self):
        return unicode(self)

    
class FactorGraph(object):
    """ Factor graph for the multi-lingual morphology induction process """

    def __init__(self):
        self.vs = {}
        self.fs = {}
        self.edges = []
    
        
    def E_step(self, iterations=10):
        """ performs the E-step """

        for i in xrange(iterations):
            pass

    def inference(self):
        # TODO: consider randommizing?
        # What's the best passing order?

        for e in self.edges:
            # does it matter that this two things are close
            # together
            #e.p2v()
            #e.p2f()
            print unicode(e)

    def brute_force_inference(self):
        """ brute force the inference for unit test """
        pass
