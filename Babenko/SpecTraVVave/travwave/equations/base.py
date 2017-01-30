#!/usr/bin/env python
# coding: utf-8
from __future__ import division

class Equation(object):
    def __init__(self, length):
        self.length = length

    def compute_kernel(self, k):
        raise NotImplementedError()

    def flux(self, u):
        raise NotImplementedError()
