""" Factor Graph for morphology """
import numpy as np
from numpy import zeros, ones, eye, dot, exp, outer
from graphviz import Graph
import itertools as it
import operator

class Edge(object):
    """ Edge """

    def __init__(self, f, v):
        # the factor
        self.f  = f
        self.m_f = .5 * ones((self.f.N))
        self.f.edges.append(self)
        # index of the edge in the variable's
        # edge list
        self.fi = len(self.f.edges)-1
        
        # the variable
        self.v = v
        self.m_v = .5 * ones((self.v.N))
        self.v.edges.append(self)
        # index of the edge in the factor's
        # edge list
        self.vi = len(self.v.edges)-1

        
    def p2f(self):
        """ Passes the message to the factor """

        stale = self.m_f
        self.m_f = self.v.b / self.m_v
        
        
    def p2v(self):
        """ Passes the message to the variable """

        if self.v.observed:
            self.m_v = self.v.b
        else:
            stale = self.m_v
            self.f.p2(self.fi)

            # update the belief by dividing out the
            # stale message and multiplying (point-wise)
            # in the new message

            self.v.b = self.v.b / stale * self.m_v
        
    def __unicode__(self):
        return unicode(self.v)+" => "+unicode(self.f)
    
        
class Variable(object):
    """ Variable in the factor graph """
    
    def __init__(self, lang, word, N):
        self.lang, self.word = lang, word
        self.N = N
        self.edges = []
        # the belief
        self.b = .5 * ones((N))
        self.observed = False
        
    def __unicode__(self):
        return unicode(self.word)+" ("+unicode(self.lang)+")"
    
    def __repr__(self):
        return unicode(self)


class ObservedVariable(Variable):
    """ Observed Variable """

    def __init__(self, lang, word, N, value):
        super(ObservedVariable, self).__init__(lang, word, N)
        assert len(value) == N
        self.observed = True
        self.b = value
    
    
class Factor(object):
    """ Factor in the factor graph """

    def __init__(self, N):
        self.b = .5 * ones((N))
        self.edges = []


        
class UnaryFactor(Factor):
    """ Unary Factor """
    
    def __init__(self, N):
        pass

    
class BinaryFactor(Factor):
    """ Binary Factor """
    
    def __init__(self, tp, source_lang, source, target_lang, target, N):
        super(BinaryFactor, self).__init__(N)
        self.tp = tp
        self.N = N
        self.source_lang, self.target_lang = source_lang, target_lang
        self.source, self.target = source, target


    def p2(self, i):
        """ pass 2 the ith factor """
        j = 0 if i == 1 else 1
        self.edges[i].m_v = exp(dot(self.edges[j].m_f, self.tp.F))
        
        
    def __unicode__(self):
        return unicode(self.source)+" ("+unicode(self.source_lang)+") -- "+unicode(self.target)+" ("+unicode(self.target_lang)+")"

    
    def __repr__(self):
        return unicode(self)

    
class FactorGraph(object):
    """ Factor graph for the multi-lingual morphology induction process """

    def __init__(self, N):
        self.N = N 
        self.vs, self.fs = {}, {}
        self.edges = []
    

    def neg(self):
        """ 
        get the negative part of the gradient.
        the expected counts
        """

        for k, f in self.fs.items():

            if len(f.edges) == 1:
                pass
            elif len(f.edges) == 2:
                e1, e2 = f.edges[0], f.edges[1]
                f.tp.neg += outer(e1.m_f, e2.m_f)
            else:
                raise("Higher-order Factor!!")

        
    def E_step(self, iterations=2):
        """ performs the E-step """

        for i in xrange(iterations):
            self.inference()

        
    def inference(self):
        # TODO: consider randommizing?
        # What's the best passing order?

        for e in self.edges:
            # does it matter that this two things are close
            # together
         
            e.p2v()
            e.p2f()
         
    def Z(self):
        """ 
        Computes the partition function based on the 
        current beliefs
        """
        Zs = []
        for k, v in self.vs.items():
            if v.observed:
                continue

            Z = 1.0
            lst = []
            for e in v.edges:
                lst.append(e.m_v)

            z = 1.0
            for l in lst:
                z *= l[0]
            Z *= (z + 1)
            z = 1.0
            for l in lst:
                z *= l[1]
            Z *= (z + 1)
            Zs.append(Z)

        return reduce(operator.mul, Zs, 1.0)
            
    def brute_force_inference(self):
        """ brute force the inference for unit test """

        counter = 0
        for k, v in self.vs.items():
            if not v.observed:
                counter += 1

        Z = 0.0
        for config_lst in it.product([0, 1], repeat=counter * self.N):
            config = np.asarray(config_lst)

            # set values
            counter = 0
            for k, v in self.vs.items():
                if not v.observed:
                    val = config[counter*self.N:(counter+1)*self.N]
                    v.b = val
                    counter += 1
            
            # get scores
            config_score = 1.0
            for k, f in self.fs.items():
                vecs = [e.v.b for e in f.edges]
                score = exp(dot(dot(vecs[0], f.tp.F), vecs[1]))
                config_score *= score

            Z += config_score
        print "Z", Z
            
    def visualize(self):
        """ visualize the factor graph as a dot file """
        dot = Graph()
        
        for k, v in self.vs.items():
            if v.observed:
                dot.node(v.word, style="filled")
            else:
                dot.node(v.word)

        for i, (k, v) in enumerate(self.fs.items()):
            dot.node(str(i), shape="square", style="bold")
            s, t = k[1], k[3]
            dot.edge(s, str(i))
            dot.edge(t, str(i))
            
        print dot.source
        #src.render('test-output/holy-grenade.gv', view=True)
