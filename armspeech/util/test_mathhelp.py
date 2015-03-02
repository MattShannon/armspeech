"""Unit tests for math helper functions."""

# Copyright 2011, 2012, 2013, 2014, 2015 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.

import unittest
import numpy as np
import numpy.linalg as la
import random
from numpy.random import randn, randint

from codedep import codeDeps

from armspeech.util import mathhelp
from armspeech.util.mathhelp import ThreshMax
from armspeech.util.mathhelp import assert_allclose
import armspeech.numpy_settings

@codeDeps()
def randBool():
    return randint(0, 2) == 0

@codeDeps()
def randLogProb():
    if randint(0, 3) == 0:
        return float('-inf')
    else:
        return 5.0 * randn()

@codeDeps(randBool)
def gen_float(allowInf = True):
    if allowInf and randint(0, 6) == 0:
        if randBool():
            return float('-inf')
        else:
            return float('inf')
    else:
        return 5.0 * randn()

@codeDeps(gen_float)
def gen_list_of_floats(length = None, allowInf = True):
    if length is None:
        length = randint(20)
    fakeLength = length * 2
    xs = [ gen_float(allowInf = allowInf) for _ in range(fakeLength) ]
    # resample to ensure some repetitions
    return [ xs[randint(fakeLength)] for _ in range(length) ]

@codeDeps()
def shapeRand(ranks = [0, 1], allDimsNonZero = False):
    rank = random.choice(ranks)
    return [ randint(1 if allDimsNonZero else 0, 10) for i in range(rank) ]

@codeDeps(ThreshMax, assert_allclose, gen_float, gen_list_of_floats,
    mathhelp.logAdd, mathhelp.logDet, mathhelp.logDetPosDef, mathhelp.logSum,
    mathhelp.reprArray, mathhelp.sampleDiscrete, randLogProb, shapeRand
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

    def test_ThreshMax(self, numPoints = 200):
        for pointIndex in range(numPoints):
            objectType = randint(3)
            if objectType == 0:
                wrap = lambda xs: xs
                value = lambda x: x
                key = None
            elif objectType == 1:
                wrap = lambda xs: [ (randint(10), x) for x in xs ]
                value = lambda x: x[1]
                key = value
            elif objectType == 2:
                wrap = lambda xs: [ (x, randint(10)) for x in xs ]
                value = lambda x: x[0]
                key = value
            else:
                assert False

            threshMaxZero = ThreshMax(0.0, key = key)
            xs = wrap(gen_list_of_floats())
            thresh = abs(randn())
            threshMax = ThreshMax(thresh, key = key)

            assert (threshMaxZero(xs) ==
                    [ x for x in xs if value(x) == max(map(key, xs)) ])

            assert threshMax(threshMax(xs)) == threshMax(xs)

            x = gen_float()
            rep = wrap([ x for _ in range(randint(10)) ])
            assert threshMax(rep) == rep

            thresh2 = abs(randn())
            thresh1 = thresh2 + abs(randn())
            threshMax1 = ThreshMax(thresh1, key = key)
            threshMax2 = ThreshMax(thresh2, key = key)
            assert threshMax2(threshMax1(xs)) == threshMax2(xs)

            ys = wrap(gen_list_of_floats())
            assert (threshMax(threshMax(xs) + threshMax(ys)) ==
                    threshMax(xs + ys))

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
                count[sample] += 1.0
            assert_allclose(count / numSamples, probs, rtol = 1e-2, atol = 1e-2)

    def test_reprArray(self, numPoints = 200):
        def evalArray(arrRepr):
            from numpy import array, zeros, dtype
            return eval(arrRepr)

        for _ in range(numPoints):
            shape = shapeRand(ranks = [0, 1, 2, 3])
            arr = randn(*shape)
            arrRepr = mathhelp.reprArray(arr)
            arrAgain = evalArray(arrRepr)
            assert_allclose(arrAgain, arr)

@codeDeps(TestMathHelp)
def suite():
    return unittest.TestLoader().loadTestsFromTestCase(TestMathHelp)

if __name__ == '__main__':
    unittest.main()
