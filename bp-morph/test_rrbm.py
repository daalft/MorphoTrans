from model import RRBM
import numpy as np
from numpy import zeros

def fd(model, EPS=0.01):
    """ finite difference check f """

    N = model.N
    f = model.f
    gW_fd, gb_fd = zeros((N, N)), zeros((N))
    for x, y in model.train:
        for i in xrange(N):
            # get gW_fd
            for j in xrange(N):
                model.W[i, j] += EPS
                val1 = f(model.W, model.b)
                model.W[i, j] -= 2*EPS
                val2 = f(model.W, model.b)
                model.W[i, j] += EPS
                gW_fd[i, j] = (val1 - val2) / (2 * EPS)
                
            # get gb_fd
            model.b[i] += EPS
            val1 = f(model.W, model.b)
            model.b[i] -= 2*EPS
            val2 = f(model.W, model.b)
            model.b[i] += EPS
            gb_fd[i] = (val1 - val2) / (2 * EPS)

        
    return gW_fd, gb_fd

N = 5
T = 10

train = []
for i in xrange(T):
    x = np.random.choice([0, 1], size=(N,), p=[1./3, 2./3])
    y = np.random.choice([0, 1], size=(N,), p=[1./3, 2./3])
    train.append((x, y))

model = RRBM(N, train)
# check partition function
assert np.allclose(model.Z(model.W, model.b, train[0][0]), model.enum_Z(model.W, model.b, train[0][0]), atol = 0.01)

gW_fd, gb_fd = fd(model)
gW, gb = model.g(model.W, model.b)

# check gradient
assert np.allclose(gW, gW_fd, atol=0.01)
assert np.allclose(gb, gb_fd, atol=0.01)

# check conversion
W2, b2 = model.vec2theta(model.theta2vec(model.W, model.b))
assert (W2 == model.W).all()
assert (b2 == model.b).all()
