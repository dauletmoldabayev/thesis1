#!/usr/bin/env python
# coding: utf-8
from __future__ import division

from .base import Equation
import numpy as np

class Whitham(Equation):
    def degree(self):
        return 2    

    def compute_kernel(self,k):
        whitham = np.zeros(len(k))
        for i in range(len(k)):        
            if k[i] == 0:          #It is incorrect))
                whitham[i] = 1
            else:    
                whitham[i]  = np.sqrt(np.tanh(k[i])/k[i])        
        return whitham
        
    def flux(self, u):
        return 0.75*u*u  

    def flux_prime(self, u):
        return 1.5*u

class Whitham_scaled(Whitham):
    def compute_kernel(self,k):
        scale = self.length/np.pi
        if k[0] == 0:
            k1 = k[1:]
            whitham = np.sqrt(scale) * np.concatenate( ([1], np.sqrt( np.tanh( k1/scale )/k1 )))
        else:    
            whitham  = np.sqrt(scale) * np.sqrt(1./k*np.tanh(1/scale * k))
       
        return whitham    

class Whitham3(Whitham):
    def degree(self):
        return 3  
        
    def flux(self, u):
        return 0.5*u**3  

    def flux_prime(self, u):
        return 1.5*u**2
        
class Whitham5(Whitham):
    def degree(self):
        return 5
          
    def flux(self, u):
        return 0.5*u**5  

    def flux_prime(self, u):
        return 2.5*u**4
        
class Whithamsqrt (Whitham):
    def degree(self):
        return 1.5
 
    def flux(self, u):
        return 2*np.power(u+1, 1.5) - 3*u - 2

    def flux_prime(self, u):
        return 3*(u+1)**(0.5)-3
    
class WhithamIce (Whitham):
     def compute_kernel(self,k):
        kappa = 0.1
        whitham = np.zeros(len(k))
        for i in range(len(k)):        
            if k[i] == 0:          #It is incorrect))
                whitham[i] = 1
            else:    
                whitham[i]  = np.sqrt(np.tanh(k[i])/k[i] * ( 1+kappa* np.power(k[i],4) ) )        
        return whitham
 
   

