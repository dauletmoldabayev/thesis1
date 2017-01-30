from __future__ import division

import numpy as np
import numba

import scipy.fftpack

def get_nodes(size, length=1.): # returns vector x \in R^N, 
    return length*(np.linspace(0, 1, size, endpoint=False) + 1/2/size)

def resample(wave, new_size):   # this function is used for navigation with different grid numbers
                                # takes in wave with fewer points and returns that wave with more points
    size = len(wave)
    new_nodes = get_nodes(new_size)
    resampled = np.interp(new_nodes, get_nodes(size), wave)
    return resampled


class Discretization(object):
    def __init__(self, equation, size): # 'equation' has class Equation (e.g. KDV)
                                        # 'size' is the number of discrete points, i.e. N
        self.equation = equation
        self.size = size

    def residual(self, u, parameters, integrconst): # parameters \in R^2, parameters = (c,a)
                                                    # u \in R^{size}, u = wave profile, i.e. u = phi
                                                    # integrconst \in R, integrconst = B
        residual = self.apply_operator(u) - parameters[0]*u + self.equation.flux(u) - integrconst
        return  residual

    def frequencies(self): # returns a vector of frequencies, k \in R^N
        return np.pi/self.equation.length*np.arange(self.size, dtype=float)

    def image(self): # returns a vector of operator's kernel applied to frequencies
                     # \alpha(k) \in R^N
        return self.equation.compute_kernel(self.frequencies())

    def bifurcation_velocity(self): # computes velocity approximation c_0
        return self.image()[1] # check this

    def get_nodes(self): # returns a vector with discrete (grid) points x_n
        nodes = get_nodes(self.size, self.equation.length)
        return nodes

    def compute_initial_guess(self, amplitude): # returns initial guess for u as a cosine function
                                                # amplitude is a small positive number 
        cosine = np.cos(get_nodes(self.size, np.pi))
        init_guess = amplitude*cosine
        return init_guess
    """
    def get_weights(self):  # this methods is not used now.
        image = self.image()
        weights = image*2/(len(image))
        weights[0] /= 2
        return weights
    """    
    def apply_operator(self, u):     # returns operator applied to u
                                     # InverseFourier( \alpha(k)* Fourier(u)(k) )
        u_ = scipy.fftpack.dct(u, norm='ortho')
        Lv = self.image() * u_
        result = scipy.fftpack.idct(Lv, norm='ortho')
        return result

"""
def _make_linear_operator(linop, weights, fik):
    \"""
    fik: array(n,k)
    linop: array(n,n) of zeros
    weights: array(k)
    \"""
    size = len(fik)
    for k in range(len(weights)):
        wk = weights[k]
        fk = fik[:,k]
        for i in range(size):
            for j in range(size):
                linop[i,j] += wk * fk[i] * fk[j]

_fast_make_linear_operator = numba.jit('void(f8[:,:], f8[:], f8[:,:])', nopython=True)(_make_linear_operator)

class DiscretizationOperator(Discretization):
    def __init__(self, *args, **kwargs):
        super(DiscretizationOperator, self).__init__(*args, **kwargs)
        self._cached_operator = {} # a dictionary containing cached linear operators

    def compute_shifted_operator(self, size, parameters):
        \"""
        Only used for testing purposes
        \"""
        return (-1)*parameters[0]*np.eye(size) + self.compute_linear_operator()

    def general_linear_operator(self, weights, nodes):
        f = np.cos
        size = len(nodes)
        ik = nodes.reshape(-1,1) * np.arange(len(weights))
        fik = f(np.pi/self.equation.length*ik) # should be replaced by a dct
        linop = np.zeros([size, size])
        _fast_make_linear_operator(linop, weights, fik)
        return linop

    def compute_linear_operator(self):
        return self.general_linear_operator(weights=self.get_weights(), nodes=self.get_nodes())

    def shifted_kernel(self):
            return np.diag(-self.bifurcation_velocity() + self.image())
"""