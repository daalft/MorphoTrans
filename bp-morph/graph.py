""" Factor Graph for morphology """

class Edge(object):
    """ Edge """

    def __init__(self, f, v):
        # the factor
        self.f  = f
        self.m_f = None
        self.f.edges.append(self)
        # index of the edge in the variable's
        # edge list
        self.fi = len(self.f.edges)-1
        
        # the variable
        self.v = v
        self.m_v = None
        self.v.edges.append(self)
        # index of the edge in the factor's
        # edge list
        self.vi = len(self.v.edges)-1

        
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

        
class Variable(object):
    """ Variable in the factor graph """
    
    def __init__(self, lang, word):
        self.lang, self.word = lang, word
        self.edges = []
        # the belief
        self.b = None

        
class Factor(object):
    """ Factor in the factor graph """

    def __init__(self):
        self.b = None

    
class UnaryFactor(Factor):
    """ Unary Factor """
    
    def __init__(self):
        pass

    
class BinaryFactor(Factor):
    """ Binary Factor """
    
    def __init__(self):
        pass

    
class FactorGraph(object):
    """ Factor graph for the multi-lingual morphology induction process """

    def __init__(self):
        self.edges = []


    def E_step(self, iterations=10):
        """ performs the E-step """

        for i in xrange(iterations):
            # TODO: consider randommizing?
            # What's the best passing order?
            for e in self.edges:

                # does it matter that this two things are close
                # together
                e.p2v()
                e.p2f()
                
