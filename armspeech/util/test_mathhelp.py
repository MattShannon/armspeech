"""Unit tests for math helper functions."""

# Copyright 2011, 2012, 2013 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.


from __future__ import division

import mathhelp
from armspeech.util.mathhelp import assert_allclose
from codedep import codeDeps

import unittest
import numpy as np
import armspeech.numpy_settings
import numpy.linalg as la
from numpy.random import randn, randint

@codeDeps()
def randLogProb():
    if randint(0, 3) == 0:
        return float('-inf')
    else:
        return 5.0 * randn()

@codeDeps(assert_allclose, mathhelp.logAdd, mathhelp.logDet, mathhelp.logSum,
    mathhelp.sampleDiscrete, randLogProb
)
class TestMathHelp(unittest.TestCase):
    def test_logAdd(self, numPoints = 200):
        for pointIndex in range(numPoints):
            a = randLogProb()
            b = randLogProb()
            r = mathhelp.logAdd(a, b)
            assert_allclose(np.exp(r), np.exp(a) + np.exp(b))

    def test_logSum(self, numPoints = 200):
        for pointIndex in range(numPoints):
            n = randint(0, 20)
            l = [ randLogProb() for _ in range(n) ]
            r = mathhelp.logSum(l)
            assert_allclose(np.exp(r), np.sum(np.exp(l)))

    def test_logDet(self, numPoints = 200):
        for pointIndex in range(numPoints):
            n = randint(0, 20)
            if randint(0, 10) == 0:
                A = np.zeros([n, n])
            else:
                A = randn(n, n)
            trueDet = la.det(A) if n > 0 else 1.0
            assert_allclose(np.exp(mathhelp.logDet(A)), abs(trueDet))

    def test_logDetPosDef(self, numPoints = 200):
        for pointIndex in range(numPoints):
            n = randint(0, 20)
            factor = randn(n, n)
            mat = np.dot(factor, factor.T)
            trueDet = la.det(mat) if n > 0 else 1.0
            assert_allclose(np.exp(mathhelp.logDetPosDef(mat)), abs(trueDet))

    def test_sampleDiscrete(self, numDists = 20, numSamples = 10000):
        for distIndex in range(numDists):
            n = randint(1, 5)
            s = 0.0
            while s == 0.0:
                probs = np.exp([ randLogProb() for _ in range(n) ])
                s = np.sum(probs)
            probs = probs / s
            assert_allclose(np.sum(probs), 1.0)
            valueProbList = list(enumerate(probs))

            count = np.zeros(n)
            for _ in xrange(numSamples):
                sample = mathhelp.sampleDiscrete(valueProbList)
                count[sample] += 1
            assert_allclose(count / numSamples, probs, rtol = 1e-2, atol = 1e-2)

@codeDeps(TestMathHelp)
def suite():
    return unittest.TestLoader().loadTestsFromTestCase(TestMathHelp)

if __name__ == '__main__':
    unittest.main()
