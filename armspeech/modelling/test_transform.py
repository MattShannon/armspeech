"""Unit tests for transforms."""

# Copyright 2011 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.


from __future__ import division

import transform as xf
from armspeech.util.mathhelp import logDet
from armspeech.util.mathhelp import assert_allclose

import unittest
import math
import random
import numpy as np
from numpy.random import randn, randint

# FIXME : add explicit tests for transform_acc

def randBool():
    return randint(0, 2) == 0

def randTag():
    return 'tag'+str(randint(0, 1000000))

def shapeRand(ranks = [0, 1]):
    rank = random.choice(ranks)
    return [ randint(0, 10) for i in range(rank) ]

def gen_genericTransform(shapeIn, shapeOut):
    # FIXME : below transforms are either linear or decompose across dimensions.
    #   Replace with something nastier? (Would provide a better test.)
    rankIn = len(shapeIn)
    rankOut = len(shapeOut)
    assert rankIn <= 1
    assert rankOut <= 1
    if (rankIn, rankOut) == (0, 0):
        return gen_genericTransform1D()
    elif (rankIn, rankOut) == (0, 1):
        # (FIXME : not a great test, since ConstantTransform has no params)
        return gen_ConstantTransform(shapeIn = shapeIn, shapeOut = shapeOut)
    elif (rankIn, rankOut) == (1, 0):
        return gen_DotProductTransform(shapeIn = shapeIn)
    else:
        if shapeIn == shapeOut and randint(0, 5) != 0:
            return xf.VectorizeTransform(gen_genericTransform1D()).withTag(randTag())
        else:
            return gen_LinearTransform(shapeIn = shapeIn, shapeOut = shapeOut)
def gen_genericInvertibleTransform(shape):
    rank = len(shape)
    assert rank <= 1
    if rank == 0:
        return gen_genericInvertibleTransform1D()
    else:
        # FIXME : below transforms are either linear or decompose across dimensions.
        #   Replace with something nastier? (Would provide a better test.)
        if randBool():
            return gen_InvertibleLinearTransform(shape = shape)
        else:
            return xf.VectorizeTransform(gen_genericInvertibleTransform1D()).withTag(randTag())
def gen_ConstantTransform(shapeIn, shapeOut):
    value = randn(*shapeOut)
    return xf.ConstantTransform(value).withTag(randTag())
def gen_DotProductTransform(shapeIn):
    assert len(shapeIn) == 1
    params = randn(*shapeIn)
    return xf.DotProductTransform(params).withTag(randTag())
def gen_LinearTransform(shapeIn, shapeOut):
    assert len(shapeIn) == 1
    assert len(shapeOut) == 1
    mat = randn(shapeIn[0], shapeOut[0])
    return xf.LinearTransform(mat).withTag(randTag())
def gen_InvertibleLinearTransform(shape):
    assert len(shape) == 1
    dim = shape[0]
    invertible = False
    while not invertible:
        mat = randn(dim, dim)
        invertible = (logDet(mat) > float('-inf'))
    return xf.LinearTransform(mat).withTag(randTag())
def gen_genericTransform1D():
    params = randn(3)
    return xf.PolynomialTransform1D(params).withTag(randTag())
def gen_genericInvertibleTransform1D():
    return gen_ScaledSinhTransform1D()
def gen_PolynomialTransform1D():
    params = randn(randint(0, 10))
    return xf.PolynomialTransform1D(params).withTag(randTag())
def gen_ScaledSinhTransform1D():
    a = randn()
    return xf.ScaledSinhTransform1D(a).withTag(randTag())
def gen_TanhTransformLogParam1D():
    params = randn(3)
    return xf.TanhTransformLogParam1D(params).withTag(randTag())
def gen_TanhTransform1D():
    params = randn(3)
    return xf.TanhTransform1D(params, warn = False).withTag(randTag())
def gen_SumTransform1D():
    return gen_InvertibleSumOfTanhLogParam1D()
def gen_InvertibleSumOfTanhLogParam1D(numTanh = 3):
    return xf.SumTransform1D([ gen_TanhTransformLogParam1D() if i > 0 else xf.IdentityTransform() for i in range(numTanh + 1) ]).withTag(randTag())
def gen_InvertibleSumOfTanh1D(numTanh = 3, tricky = False):
    """
    if tricky is set at least one component transform will have negative derivative
    (the overall sum still has positive derivative everywhere)
    """
    while True:
        transforms = [ gen_TanhTransform1D() for i in range(numTanh) ]
        if not tricky or numTanh == 0 or any([ transform.a * transform.b < 0.0 for transform in transforms ]):
            derivLowerBound = 1.0 + sum([ min(transform.a * transform.b, 0.0) for transform in transforms ])
            if derivLowerBound > 0.0:
                return xf.SumTransform1D([xf.IdentityTransform()] + transforms).withTag(randTag())
def gen_SumOfTanh1D(numTanh = 3):
    return xf.SumTransform1D([ gen_TanhTransform1D() if i > 0 else xf.IdentityTransform() for i in range(numTanh + 1) ]).withTag(randTag())

def check_deriv(transform, x, eps):
    delta = randn(*np.shape(x)) * eps
    numericDelta = transform(x + delta) - transform(x)
    analyticDelta = np.dot(delta, transform.deriv(x))
    assert_allclose(numericDelta, analyticDelta, rtol = 1e-4)
def check_derivDeriv(transform, x, eps):
    assert np.shape(x) == ()
    delta = randn() * eps
    numericDelta = transform.deriv(x + delta) - transform.deriv(x)
    analyticDelta = np.dot(delta, transform.derivDeriv(x))
    assert_allclose(numericDelta, analyticDelta, rtol = 1e-4)
def check_derivParams(transform, x, eps):
    params = transform.params
    paramsDelta = randn(*np.shape(params)) * eps
    numericDelta = transform.parseAll(params + paramsDelta)(x) - transform(x)
    analyticDelta = np.dot(paramsDelta, transform.derivParams(x))
    assert_allclose(numericDelta, analyticDelta, rtol = 1e-4)
def check_derivParamsDeriv(transform, x, eps):
    assert np.shape(x) == ()
    delta = randn() * eps
    numericDelta = transform.derivParams(x + delta) - transform.derivParams(x)
    analyticDelta = np.dot(delta, transform.derivParamsDeriv(x))
    assert_allclose(numericDelta, analyticDelta, rtol = 1e-4)
def computeLogJac(transform, x):
    shapeOut = np.shape(x)
    deriv = transform.deriv(x)
    if len(shapeOut) == 0:
        return math.log(abs(deriv))
    elif len(shapeOut) == 1:
        return logDet(deriv)
    else:
        raise RuntimeError('log-Jacobian computation not implemented for output of rank >= 2')
def check_logJac(transform, x, eps):
    numericLJ = computeLogJac(transform, x)
    analyticLJ = transform.logJac(x)
    assert np.shape(analyticLJ) == ()
    assert_allclose(numericLJ, analyticLJ, atol = 1e-10)
def check_logJacDeriv(transform, x, eps):
    delta = randn(*np.shape(x)) * eps
    numericDelta = transform.logJac(x + delta) - transform.logJac(x)
    analyticDelta = np.dot(delta, transform.logJacDeriv(x))
    assert_allclose(numericDelta, analyticDelta, atol = 1e-10, rtol = 1e-4)
def check_logJacDerivParams(transform, x, eps):
    params = transform.params
    paramsDelta = randn(*np.shape(params)) * eps
    numericDeltaLJ = transform.parseAll(params + paramsDelta).logJac(x) - transform.logJac(x)
    analyticDeltaLJ = np.dot(transform.logJacDerivParams(x), paramsDelta)
    assert_allclose(numericDeltaLJ, analyticDeltaLJ, atol = 1e-10, rtol = 1e-4)
def check_inv(transform, x, y):
    """(N.B. x and y not supposed to correspond to each other)"""
    xAgain = transform.inv(transform(x))
    assert_allclose(xAgain, x, msg = 'inverse not consistent')
    yAgain = transform(transform.inv(y))
    assert_allclose(yAgain, y, msg = 'inverse not consistent')

def checkTransform(transform, shapeIn, invertible, hasParams, is1D, eps, its, checkAdditional = None):
    assert transform.tag is not None
    transformEvaled = xf.eval_local(repr(transform))
    assert transformEvaled.tag == transform.tag
    assert repr(transform) == repr(transformEvaled)
    if hasParams:
        params = transform.params
        transformParsed = transform.parseAll(params)
        assert transformParsed.tag == transform.tag
        assert_allclose(transformParsed.params, params)
        assert_allclose(transformEvaled.params, params, rtol = 1e-5)
    for it in range(its):
        x = randn(*shapeIn)
        if checkAdditional is not None:
            checkAdditional(transform, x, eps)
        if True:
            assert_allclose(transformEvaled(x), transform(x), rtol = 5e-4)
        if hasParams:
            assert_allclose(transformParsed(x), transform(x))
        if True:
            check_deriv(transform, x, eps)
        if is1D:
            check_derivDeriv(transform, x, eps)
        if hasParams:
            check_derivParams(transform, x, eps)
        if hasParams and is1D:
            check_derivParamsDeriv(transform, x, eps)
        if invertible:
            check_logJac(transform, x, eps)
        if invertible:
            check_logJacDeriv(transform, x, eps)
        if hasParams and invertible:
            check_logJacDerivParams(transform, x, eps)
        if invertible:
            check_inv(transform, x, randn(*shapeIn))

class TestTransform(unittest.TestCase):
    def test_ConstantTransform(self, eps = 1e-8, its = 10, itsPerTransform = 10):
        for it in range(its):
            shapeIn = shapeRand()
            shapeOut = shapeRand()
            axf = gen_ConstantTransform(shapeIn = shapeIn, shapeOut = shapeOut)
            checkTransform(axf, shapeIn, invertible = False, hasParams = False, is1D = False, eps = eps, its = itsPerTransform)
    def test_IdentityTransform(self, eps = 1e-8, its = 10, itsPerTransform = 10):
        def checkAdditional(transform, x, eps):
            assert_allclose(transform(x), x)
        for it in range(its):
            axf = xf.IdentityTransform().withTag(randTag())
            shapeIn = shapeRand()
            checkTransform(axf, shapeIn, invertible = True, hasParams = True, is1D = False, eps = eps, its = itsPerTransform, checkAdditional = checkAdditional)
    def test_DotProductTransform(self, eps = 1e-8, its = 10, itsPerTransform = 10):
        for it in range(its):
            shapeIn = shapeRand(ranks = [1])
            axf = gen_DotProductTransform(shapeIn = shapeIn)
            checkTransform(axf, shapeIn, invertible = False, hasParams = True, is1D = False, eps = eps, its = itsPerTransform)
    def test_LinearTransform(self, eps = 1e-8, its = 10, itsPerTransform = 10):
        for it in range(its):
            shapeIn = shapeRand(ranks = [1])
            shapeOut = shapeRand(ranks = [1])
            axf = gen_LinearTransform(shapeIn = shapeIn, shapeOut = shapeOut)
            checkTransform(axf, shapeIn, invertible = False, hasParams = True, is1D = False, eps = eps, its = itsPerTransform)
            shape = shapeRand(ranks = [1])
            axf = gen_InvertibleLinearTransform(shape = shape)
            checkTransform(axf, shape, invertible = True, hasParams = True, is1D = False, eps = eps, its = itsPerTransform)
    def test_FrozenTransform(self, eps = 1e-8, its = 10, itsPerTransform = 10):
        for it in range(its):
            shapeIn = shapeRand()
            if randBool():
                shapeOut = shapeIn
            else:
                shapeOut = shapeRand()
            axfSub = gen_genericTransform(shapeIn = shapeIn, shapeOut = shapeOut)
            axf = xf.FrozenTransform(axfSub).withTag(randTag())
            checkTransform(axf, shapeIn, invertible = False, hasParams = False, is1D = False, eps = eps, its = itsPerTransform)
            shape = shapeRand()
            axfSub = gen_genericInvertibleTransform(shape = shape)
            axf = xf.FrozenTransform(axfSub).withTag(randTag())
            checkTransform(axf, shape, invertible = True, hasParams = False, is1D = False, eps = eps, its = itsPerTransform)
    def test_InvertedTransform(self, eps = 1e-8, its = 10, itsPerTransform = 10):
        for it in range(its):
            shape = shapeRand()
            axfSub = gen_genericInvertibleTransform(shape = shape)
            axf = xf.InvertedTransform(axfSub).withTag(randTag())
            checkTransform(axf, shape, invertible = True, hasParams = True, is1D = False, eps = eps, its = itsPerTransform)
    def test_AddBias(self, eps = 1e-8, its = 10, itsPerTransform = 10):
        for it in range(its):
            axf = xf.AddBias().withTag(randTag())
            shapeIn = [randint(0, 10)]
            checkTransform(axf, shapeIn, invertible = False, hasParams = False, is1D = False, eps = eps, its = itsPerTransform)
    def test_MinusPrev(self, eps = 1e-8, its = 10, itsPerTransform = 10):
        for it in range(its):
            axf = xf.MinusPrev().withTag(randTag())
            shapeIn = [randint(1, 10)]
            checkTransform(axf, shapeIn, invertible = False, hasParams = False, is1D = False, eps = eps, its = itsPerTransform)
    # FIXME : add test for Msd01ToVector
    def test_VectorizeTransform(self, eps = 1e-8, its = 10, itsPerTransform = 10):
        def checkAdditional(transform, x, eps):
            assert_allclose(transform(x), np.array(map(transform.transform1D, x)))
        for it in range(its):
            F = xf.VectorizeTransform(gen_genericTransform1D()).withTag(randTag())
            shapeIn = shapeRand([1])
            checkTransform(F, shapeIn, invertible = False, hasParams = True, is1D = False, eps = eps, its = itsPerTransform, checkAdditional = checkAdditional)
            F = xf.VectorizeTransform(gen_genericInvertibleTransform1D()).withTag(randTag())
            shapeIn = shapeRand([1])
            checkTransform(F, shapeIn, invertible = True, hasParams = True, is1D = False, eps = eps, its = itsPerTransform, checkAdditional = checkAdditional)
    def test_PolynomialTransform1D(self, eps = 1e-8, its = 50, itsPerTransform = 10):
        for it in range(its):
            f = gen_PolynomialTransform1D()
            checkTransform(f, [], invertible = False, hasParams = True, is1D = True, eps = eps, its = itsPerTransform)
    def test_ScaledSinhTransform1D(self, eps = 1e-8, its = 50, itsPerTransform = 10):
        for it in range(its):
            f = gen_ScaledSinhTransform1D()
            checkTransform(f, [], invertible = True, hasParams = True, is1D = True, eps = eps, its = itsPerTransform)
    def test_TanhTransformLogParam1D(self, eps = 1e-8, its = 10, itsPerTransform = 10):
        for it in range(its):
            f = gen_TanhTransformLogParam1D()
            checkTransform(f, [], invertible = False, hasParams = True, is1D = True, eps = eps, its = itsPerTransform)
    def test_TanhTransform1D(self, eps = 1e-8, its = 10, itsPerTransform = 10):
        for it in range(its):
            f = gen_TanhTransform1D()
            checkTransform(f, [], invertible = False, hasParams = True, is1D = True, eps = eps, its = itsPerTransform)
    def test_SumTransform1D(self, eps = 1e-8, its = 10, itsPerTransform = 10):
        for it in range(its):
            numTanh = randint(0, 5)
            f = gen_InvertibleSumOfTanhLogParam1D(numTanh)
            checkTransform(f, [], invertible = True, hasParams = True, is1D = True, eps = eps, its = itsPerTransform)
            f = gen_SumOfTanh1D(numTanh)
            checkTransform(f, [], invertible = False, hasParams = True, is1D = True, eps = eps, its = itsPerTransform)
            f = gen_InvertibleSumOfTanh1D(numTanh, tricky = True)
            checkTransform(f, [], invertible = True, hasParams = True, is1D = True, eps = eps, its = itsPerTransform)

def gen_genericOutputTransform(shapeInput, shapeOutput):
    # FIXME : something more general?
    return gen_ShiftOutputTransform(shapeInput = shapeInput, shapeOutput = shapeOutput)
def gen_SimpleOutputTransform(shapeInput, shapeOutput):
    axf = gen_genericInvertibleTransform(shape = shapeOutput)
    checkDerivPositive1D = (len(shapeOutput) == 0 and randBool())
    return xf.SimpleOutputTransform(axf, checkDerivPositive1D = checkDerivPositive1D).withTag(randTag())
def gen_ShiftOutputTransform(shapeInput, shapeOutput):
    axf = gen_genericTransform(shapeIn = shapeInput, shapeOut = shapeOutput)
    return xf.ShiftOutputTransform(axf).withTag(randTag())

def check_derivInput(outputTransform, input, x, eps):
    delta = randn(*np.shape(input)) * eps
    numericDelta = outputTransform(input + delta, x) - outputTransform(input, x)
    analyticDelta = np.dot(delta, outputTransform.derivInput(input, x))
    assert_allclose(numericDelta, analyticDelta, rtol = 1e-4)
def check_logJacDerivInput(outputTransform, input, x, eps):
    delta = randn(*np.shape(input)) * eps
    numericDelta = outputTransform.logJac(input + delta, x) - outputTransform.logJac(input, x)
    analyticDelta = np.dot(delta, outputTransform.logJacDerivInput(input, x))
    assert_allclose(numericDelta, analyticDelta, atol = 1e-10, rtol = 1e-4)

def checkOutputTransform(outputTransform, shapeInput, shapeOutput, hasParams, eps, its, checkAdditional = None):
    outputTransformEvaled = xf.eval_local(repr(outputTransform))
    assert repr(outputTransform) == repr(outputTransformEvaled)
    if hasParams:
        params = outputTransform.params
        outputTransformParsed = outputTransform.parseAll(params)
        assert_allclose(outputTransformParsed.params, params)
        assert_allclose(outputTransformEvaled.params, params, rtol = 1e-5)
    for it in range(its):
        input = randn(*shapeInput)
        x = randn(*shapeOutput)
        transform = outputTransform.atInput(input)
        if checkAdditional is not None:
            checkAdditional(outputTransform, input, x, eps)
        if True:
            assert_allclose(outputTransform(input, x), transform(x))
        if True:
            assert_allclose(outputTransformEvaled(input, x), outputTransform(input, x), rtol = 5e-4)
        if hasParams:
            assert_allclose(outputTransformParsed(input, x), outputTransform(input, x))
        if True:
            check_deriv(transform, x, eps)
        if hasParams:
            check_derivParams(transform, x, eps)
        if True:
            check_logJac(transform, x, eps)
        if True:
            check_logJacDeriv(transform, x, eps)
        if hasParams:
            check_logJacDerivParams(transform, x, eps)
        if True:
            check_inv(transform, x, randn(*shapeOutput))

class TestOutputTransform(unittest.TestCase):
    def test_SimpleOutputTransform(self, eps = 1e-8, its = 10, itsPerTransform = 10):
        def checkAdditional(outputTransform, input, x, eps):
            assert_allclose(outputTransform(input, x), outputTransform.transform(x))
        for it in range(its):
            shapeInput = shapeRand()
            shapeOutput = shapeRand()
            axf = gen_SimpleOutputTransform(shapeInput = shapeInput, shapeOutput = shapeOutput)
            checkOutputTransform(axf, shapeInput, shapeOutput, hasParams = True, eps = eps, its = itsPerTransform, checkAdditional = checkAdditional)
    def test_checkDerivPositive1D_for_SimpleOutputTransform(self):
        outputTransform = xf.SimpleOutputTransform(xf.TanhTransform1D([1.0, 0.5, 0.0]), checkDerivPositive1D = True)
        assert outputTransform.deriv([], 0.0) > 0.0
        outputTransform = xf.SimpleOutputTransform(xf.TanhTransform1D([-1.0, 0.5, 0.0]), checkDerivPositive1D = True)
        self.assertRaises(xf.DerivativeNotPositiveError, outputTransform.deriv, [], 0.0)
    def test_ShiftOutputTransform(self, eps = 1e-8, its = 10, itsPerTransform = 10):
        def checkAdditional(outputTransform, input, x, eps):
            assert_allclose(outputTransform(input, x), x + outputTransform.shift(input))
        for it in range(its):
            shapeInput = shapeRand()
            if randBool():
                shapeOutput = shapeInput
            else:
                shapeOutput = shapeRand()
            axf = gen_ShiftOutputTransform(shapeInput = shapeInput, shapeOutput = shapeOutput)
            checkOutputTransform(axf, shapeInput, shapeOutput, hasParams = True, eps = eps, its = itsPerTransform, checkAdditional = checkAdditional)

def suite():
    return unittest.TestSuite([
        unittest.TestLoader().loadTestsFromTestCase(TestTransform),
        unittest.TestLoader().loadTestsFromTestCase(TestOutputTransform),
    ])

if __name__ == '__main__':
    unittest.main()
