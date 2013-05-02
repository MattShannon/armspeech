"""Unit tests for transforms."""

# Copyright 2011, 2012, 2013 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.


from __future__ import division

import transform as xf
from armspeech.util.mathhelp import logDet
from armspeech.util.mathhelp import assert_allclose
from codedep import codeDeps, ForwardRef

import unittest
import math
import random
import numpy as np
import armspeech.numpy_settings
from numpy.random import randn, randint

# FIXME : add explicit tests for transform_acc

@codeDeps()
def randBool():
    return randint(0, 2) == 0

@codeDeps()
def randTag():
    return 'tag'+str(randint(0, 1000000))

@codeDeps()
def shapeRand(ranks = [0, 1], allDimsNonZero = False):
    rank = random.choice(ranks)
    return [ randint(1 if allDimsNonZero else 0, 10) for i in range(rank) ]

@codeDeps(ForwardRef(lambda: gen_ConstantTransform),
    ForwardRef(lambda: gen_DotProductTransform),
    ForwardRef(lambda: gen_LinearTransform),
    ForwardRef(lambda: gen_genericTransform1D), randTag, xf.VectorizeTransform
)
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
@codeDeps(ForwardRef(lambda: gen_InvertibleLinearTransform),
    ForwardRef(lambda: gen_genericInvertibleTransform1D), randBool, randTag,
    xf.VectorizeTransform
)
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
@codeDeps(randTag, xf.ConstantTransform)
def gen_ConstantTransform(shapeIn, shapeOut):
    value = randn(*shapeOut)
    return xf.ConstantTransform(value).withTag(randTag())
@codeDeps(randTag, xf.DotProductTransform)
def gen_DotProductTransform(shapeIn):
    assert len(shapeIn) == 1
    params = randn(*shapeIn)
    return xf.DotProductTransform(params).withTag(randTag())
@codeDeps(randTag, xf.LinearTransform)
def gen_LinearTransform(shapeIn, shapeOut):
    assert len(shapeIn) == 1
    assert len(shapeOut) == 1
    mat = randn(shapeIn[0], shapeOut[0])
    return xf.LinearTransform(mat).withTag(randTag())
@codeDeps(logDet, randTag, xf.LinearTransform)
def gen_InvertibleLinearTransform(shape):
    assert len(shape) == 1
    dim = shape[0]
    invertible = False
    while not invertible:
        mat = randn(dim, dim)
        invertible = (logDet(mat) > float('-inf'))
    return xf.LinearTransform(mat).withTag(randTag())
@codeDeps(randTag, xf.PolynomialTransform1D)
def gen_genericTransform1D():
    params = randn(3)
    return xf.PolynomialTransform1D(params).withTag(randTag())
@codeDeps(ForwardRef(lambda: gen_ScaledSinhTransform1D))
def gen_genericInvertibleTransform1D():
    return gen_ScaledSinhTransform1D()
@codeDeps(randTag, xf.PolynomialTransform1D)
def gen_PolynomialTransform1D():
    params = randn(randint(0, 10))
    return xf.PolynomialTransform1D(params).withTag(randTag())
@codeDeps(randTag, xf.ScaledSinhTransform1D)
def gen_ScaledSinhTransform1D():
    a = randn()
    return xf.ScaledSinhTransform1D(a).withTag(randTag())
@codeDeps(randTag, xf.TanhTransformLogParam1D)
def gen_TanhTransformLogParam1D():
    params = randn(3)
    return xf.TanhTransformLogParam1D(params).withTag(randTag())
@codeDeps(randTag, xf.TanhTransform1D)
def gen_TanhTransform1D():
    params = randn(3)
    return xf.TanhTransform1D(params, warn = False).withTag(randTag())
@codeDeps(ForwardRef(lambda: gen_InvertibleSumOfTanhLogParam1D))
def gen_SumTransform1D():
    return gen_InvertibleSumOfTanhLogParam1D()
@codeDeps(gen_TanhTransformLogParam1D, randTag, xf.IdentityTransform,
    xf.SumTransform1D
)
def gen_InvertibleSumOfTanhLogParam1D(numTanh = 3):
    return xf.SumTransform1D([ gen_TanhTransformLogParam1D() if i > 0 else xf.IdentityTransform() for i in range(numTanh + 1) ]).withTag(randTag())
@codeDeps(gen_TanhTransform1D, randTag, xf.IdentityTransform, xf.SumTransform1D)
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
@codeDeps(gen_TanhTransform1D, randTag, xf.IdentityTransform, xf.SumTransform1D)
def gen_SumOfTanh1D(numTanh = 3):
    return xf.SumTransform1D([ gen_TanhTransform1D() if i > 0 else xf.IdentityTransform() for i in range(numTanh + 1) ]).withTag(randTag())

@codeDeps(assert_allclose)
def check_deriv(transform, x, eps):
    direction = randn(*np.shape(x))
    numericDeriv = (transform(x + direction * eps) - transform(x)) / eps
    analyticDeriv = np.dot(direction, transform.deriv(x))
    assert_allclose(numericDeriv, analyticDeriv, atol = 1e-6, rtol = 1e-4)
@codeDeps(assert_allclose)
def check_derivDeriv(transform, x, eps):
    assert np.shape(x) == ()
    direction = randn()
    numericDeriv = (transform.deriv(x + direction * eps) - transform.deriv(x)) / eps
    analyticDeriv = np.dot(direction, transform.derivDeriv(x))
    assert_allclose(numericDeriv, analyticDeriv, atol = 1e-6, rtol = 1e-4)
@codeDeps(assert_allclose)
def check_derivParams(transform, x, eps):
    params = transform.params
    paramsDirection = randn(*np.shape(params))
    numericDeriv = (transform.parseAll(params + paramsDirection * eps)(x) - transform(x)) / eps
    analyticDeriv = np.dot(paramsDirection, transform.derivParams(x))
    assert_allclose(numericDeriv, analyticDeriv, atol = 1e-6, rtol = 1e-4)
@codeDeps(assert_allclose)
def check_derivParamsDeriv(transform, x, eps):
    assert np.shape(x) == ()
    direction = randn()
    numericDeriv = (transform.derivParams(x + direction * eps) - transform.derivParams(x)) / eps
    analyticDeriv = np.dot(direction, transform.derivParamsDeriv(x))
    assert_allclose(numericDeriv, analyticDeriv, atol = 1e-6, rtol = 1e-4)
@codeDeps(logDet)
def computeLogJac(transform, x):
    shapeOut = np.shape(x)
    deriv = transform.deriv(x)
    if len(shapeOut) == 0:
        return math.log(abs(deriv))
    elif len(shapeOut) == 1:
        return logDet(deriv)
    else:
        raise RuntimeError('log-Jacobian computation not implemented for output of rank >= 2')
@codeDeps(assert_allclose, computeLogJac)
def check_logJac(transform, x):
    numericLJ = computeLogJac(transform, x)
    analyticLJ = transform.logJac(x)
    assert np.shape(analyticLJ) == ()
    assert_allclose(numericLJ, analyticLJ, atol = 1e-10)
@codeDeps(assert_allclose)
def check_logJacDeriv(transform, x, eps):
    direction = randn(*np.shape(x))
    numericDeriv = (transform.logJac(x + direction * eps) - transform.logJac(x)) / eps
    analyticDeriv = np.dot(direction, transform.logJacDeriv(x))
    assert_allclose(numericDeriv, analyticDeriv, atol = 1e-5, rtol = 1e-4)
@codeDeps(assert_allclose)
def check_logJacDerivParams(transform, x, eps):
    params = transform.params
    paramsDirection = randn(*np.shape(params))
    numericDerivLJ = (transform.parseAll(params + paramsDirection * eps).logJac(x) - transform.logJac(x)) / eps
    analyticDerivLJ = np.dot(transform.logJacDerivParams(x), paramsDirection)
    assert_allclose(numericDerivLJ, analyticDerivLJ, atol = 1e-6, rtol = 1e-4)
@codeDeps(assert_allclose)
def check_inv(transform, x, y):
    """(N.B. x and y not supposed to correspond to each other)"""
    xAgain = transform.inv(transform(x))
    assert_allclose(xAgain, x, msg = 'inverse not consistent')
    yAgain = transform(transform.inv(y))
    assert_allclose(yAgain, y, msg = 'inverse not consistent')

@codeDeps(assert_allclose, check_deriv, check_derivDeriv, check_derivParams,
    check_derivParamsDeriv, check_inv, check_logJac, check_logJacDeriv,
    check_logJacDerivParams, xf.eval_local
)
def checkTransform(transform, shapeIn, invertible, hasDeriv, hasParams, is1D, eps, its, checkAdditional = None):
    assert transform.tag is not None
    transformEvaled = xf.eval_local(repr(transform))
    assert transformEvaled.tag == transform.tag
    assert repr(transform) == repr(transformEvaled)
    if hasParams:
        params = transform.params
        transformParsed = transform.parseAll(params)
        assert transformParsed.tag == transform.tag
        assert_allclose(transformParsed.params, params)
        assert_allclose(transformEvaled.params, params)
    for it in range(its):
        x = randn(*shapeIn)
        if checkAdditional is not None:
            checkAdditional(transform, x, eps)
        if True:
            assert_allclose(transformEvaled(x), transform(x))
        if hasParams:
            assert_allclose(transformParsed(x), transform(x))
        if hasDeriv:
            check_deriv(transform, x, eps)
        if is1D and hasDeriv:
            check_derivDeriv(transform, x, eps)
        if hasParams:
            check_derivParams(transform, x, eps)
        if hasDeriv and hasParams and is1D:
            check_derivParamsDeriv(transform, x, eps)
        if invertible and hasDeriv:
            check_logJac(transform, x)
        if invertible and hasDeriv:
            check_logJacDeriv(transform, x, eps)
        if invertible and hasDeriv and hasParams:
            check_logJacDerivParams(transform, x, eps)
        if invertible:
            check_inv(transform, x, randn(*shapeIn))

@codeDeps(assert_allclose, checkTransform, gen_ConstantTransform,
    gen_DotProductTransform, gen_InvertibleLinearTransform,
    gen_InvertibleSumOfTanh1D, gen_InvertibleSumOfTanhLogParam1D,
    gen_LinearTransform, gen_PolynomialTransform1D, gen_ScaledSinhTransform1D,
    gen_SumOfTanh1D, gen_TanhTransform1D, gen_TanhTransformLogParam1D,
    gen_genericInvertibleTransform, gen_genericInvertibleTransform1D,
    gen_genericTransform, gen_genericTransform1D, randBool, randTag, shapeRand,
    xf.AddBias, xf.AddBiasVec, xf.FrozenTransform, xf.IdentityTransform,
    xf.InvertedTransform, xf.MinusPrev, xf.TransposeTransform,
    xf.VectorizeTransform
)
class TestTransform(unittest.TestCase):
    def test_ConstantTransform(self, eps = 1e-8, its = 10, itsPerTransform = 10):
        for it in range(its):
            shapeIn = shapeRand()
            shapeOut = shapeRand()
            axf = gen_ConstantTransform(shapeIn = shapeIn, shapeOut = shapeOut)
            checkTransform(axf, shapeIn, invertible = False, hasDeriv = True, hasParams = False, is1D = False, eps = eps, its = itsPerTransform)
    def test_IdentityTransform(self, eps = 1e-8, its = 10, itsPerTransform = 10):
        def checkAdditional(transform, x, eps):
            assert_allclose(transform(x), x)
        for it in range(its):
            axf = xf.IdentityTransform().withTag(randTag())
            shapeIn = shapeRand()
            checkTransform(axf, shapeIn, invertible = True, hasDeriv = True, hasParams = True, is1D = False, eps = eps, its = itsPerTransform, checkAdditional = checkAdditional)
    def test_DotProductTransform(self, eps = 1e-8, its = 10, itsPerTransform = 10):
        for it in range(its):
            shapeIn = shapeRand(ranks = [1])
            axf = gen_DotProductTransform(shapeIn = shapeIn)
            checkTransform(axf, shapeIn, invertible = False, hasDeriv = True, hasParams = True, is1D = False, eps = eps, its = itsPerTransform)
    def test_LinearTransform(self, eps = 1e-8, its = 10, itsPerTransform = 10):
        for it in range(its):
            shapeIn = shapeRand(ranks = [1])
            shapeOut = shapeRand(ranks = [1])
            axf = gen_LinearTransform(shapeIn = shapeIn, shapeOut = shapeOut)
            checkTransform(axf, shapeIn, invertible = False, hasDeriv = True, hasParams = True, is1D = False, eps = eps, its = itsPerTransform)
            shape = shapeRand(ranks = [1])
            axf = gen_InvertibleLinearTransform(shape = shape)
            checkTransform(axf, shape, invertible = True, hasDeriv = True, hasParams = True, is1D = False, eps = eps, its = itsPerTransform)
    def test_FrozenTransform(self, eps = 1e-8, its = 10, itsPerTransform = 10):
        for it in range(its):
            shapeIn = shapeRand()
            if randBool():
                shapeOut = shapeIn
            else:
                shapeOut = shapeRand()
            axfSub = gen_genericTransform(shapeIn = shapeIn, shapeOut = shapeOut)
            axf = xf.FrozenTransform(axfSub).withTag(randTag())
            checkTransform(axf, shapeIn, invertible = False, hasDeriv = True, hasParams = False, is1D = False, eps = eps, its = itsPerTransform)
            shape = shapeRand()
            axfSub = gen_genericInvertibleTransform(shape = shape)
            axf = xf.FrozenTransform(axfSub).withTag(randTag())
            checkTransform(axf, shape, invertible = True, hasDeriv = True, hasParams = False, is1D = False, eps = eps, its = itsPerTransform)
    def test_InvertedTransform(self, eps = 1e-8, its = 10, itsPerTransform = 10):
        for it in range(its):
            shape = shapeRand()
            axfSub = gen_genericInvertibleTransform(shape = shape)
            axf = xf.InvertedTransform(axfSub).withTag(randTag())
            checkTransform(axf, shape, invertible = True, hasDeriv = True, hasParams = True, is1D = False, eps = eps, its = itsPerTransform)
    def test_TransposeTransform(self, eps = 1e-8, its = 10, itsPerTransform = 10):
        for it in range(its):
            axf = xf.TransposeTransform().withTag(randTag())
            shapeIn = shapeRand([2, 3, 4, 5], allDimsNonZero = True)
            checkTransform(axf, shapeIn, invertible = True, hasDeriv = False, hasParams = False, is1D = False, eps = eps, its = itsPerTransform)
    def test_AddBias(self, eps = 1e-8, its = 10, itsPerTransform = 10):
        for it in range(its):
            axf = xf.AddBias().withTag(randTag())
            shapeIn = [randint(0, 10)]
            checkTransform(axf, shapeIn, invertible = False, hasDeriv = True, hasParams = False, is1D = False, eps = eps, its = itsPerTransform)
    def test_AddBiasVec(self, eps = 1e-8, its = 10, itsPerTransform = 10):
        for it in range(its):
            axf = xf.AddBiasVec().withTag(randTag())
            shapeIn = [randint(0, 10), randint(0, 10)]
            # FIXME : implement deriv and enable hasDeriv
            checkTransform(axf, shapeIn, invertible = False, hasDeriv = False, hasParams = False, is1D = False, eps = eps, its = itsPerTransform)
    def test_MinusPrev(self, eps = 1e-8, its = 10, itsPerTransform = 10):
        for it in range(its):
            axf = xf.MinusPrev().withTag(randTag())
            shapeIn = [randint(1, 10)]
            checkTransform(axf, shapeIn, invertible = False, hasDeriv = True, hasParams = False, is1D = False, eps = eps, its = itsPerTransform)
    # FIXME : add test for Msd01ToVector
    def test_VectorizeTransform(self, eps = 1e-8, its = 10, itsPerTransform = 10):
        def checkAdditional(transform, x, eps):
            assert_allclose(transform(x), np.array(map(transform.transform1D, x)))
        for it in range(its):
            F = xf.VectorizeTransform(gen_genericTransform1D()).withTag(randTag())
            shapeIn = shapeRand([1])
            checkTransform(F, shapeIn, invertible = False, hasDeriv = True, hasParams = True, is1D = False, eps = eps, its = itsPerTransform, checkAdditional = checkAdditional)
            F = xf.VectorizeTransform(gen_genericInvertibleTransform1D()).withTag(randTag())
            shapeIn = shapeRand([1])
            checkTransform(F, shapeIn, invertible = True, hasDeriv = True, hasParams = True, is1D = False, eps = eps, its = itsPerTransform, checkAdditional = checkAdditional)
    def test_PolynomialTransform1D(self, eps = 1e-8, its = 50, itsPerTransform = 10):
        for it in range(its):
            f = gen_PolynomialTransform1D()
            checkTransform(f, [], invertible = False, hasDeriv = True, hasParams = True, is1D = True, eps = eps, its = itsPerTransform)
    def test_ScaledSinhTransform1D(self, eps = 1e-8, its = 50, itsPerTransform = 10):
        for it in range(its):
            f = gen_ScaledSinhTransform1D()
            checkTransform(f, [], invertible = True, hasDeriv = True, hasParams = True, is1D = True, eps = eps, its = itsPerTransform)
    def test_TanhTransformLogParam1D(self, eps = 1e-8, its = 10, itsPerTransform = 10):
        for it in range(its):
            f = gen_TanhTransformLogParam1D()
            checkTransform(f, [], invertible = False, hasDeriv = True, hasParams = True, is1D = True, eps = eps, its = itsPerTransform)
    def test_TanhTransform1D(self, eps = 1e-8, its = 10, itsPerTransform = 10):
        for it in range(its):
            f = gen_TanhTransform1D()
            checkTransform(f, [], invertible = False, hasDeriv = True, hasParams = True, is1D = True, eps = eps, its = itsPerTransform)
    def test_SumTransform1D(self, eps = 1e-8, its = 10, itsPerTransform = 10):
        for it in range(its):
            numTanh = randint(0, 5)
            f = gen_InvertibleSumOfTanhLogParam1D(numTanh)
            checkTransform(f, [], invertible = True, hasDeriv = True, hasParams = True, is1D = True, eps = eps, its = itsPerTransform)
            f = gen_SumOfTanh1D(numTanh)
            checkTransform(f, [], invertible = False, hasDeriv = True, hasParams = True, is1D = True, eps = eps, its = itsPerTransform)
            f = gen_InvertibleSumOfTanh1D(numTanh, tricky = True)
            checkTransform(f, [], invertible = True, hasDeriv = True, hasParams = True, is1D = True, eps = eps, its = itsPerTransform)

@codeDeps(ForwardRef(lambda: gen_ShiftOutputTransform))
def gen_genericOutputTransform(shapeInput, shapeOutput):
    # FIXME : something more general?
    return gen_ShiftOutputTransform(shapeInput = shapeInput, shapeOutput = shapeOutput)
@codeDeps(gen_genericInvertibleTransform, randBool, randTag,
    xf.SimpleOutputTransform
)
def gen_SimpleOutputTransform(shapeInput, shapeOutput):
    axf = gen_genericInvertibleTransform(shape = shapeOutput)
    checkDerivPositive1D = (len(shapeOutput) == 0 and randBool())
    return xf.SimpleOutputTransform(axf, checkDerivPositive1D = checkDerivPositive1D).withTag(randTag())
@codeDeps(gen_genericTransform, randTag, xf.ShiftOutputTransform)
def gen_ShiftOutputTransform(shapeInput, shapeOutput):
    axf = gen_genericTransform(shapeIn = shapeInput, shapeOut = shapeOutput)
    return xf.ShiftOutputTransform(axf).withTag(randTag())

@codeDeps(assert_allclose)
def check_derivInput(outputTransform, input, x, eps):
    direction = randn(*np.shape(input))
    numericDeriv = (outputTransform(input + direction * eps, x) - outputTransform(input, x)) / eps
    analyticDeriv = np.dot(direction, outputTransform.derivInput(input, x))
    assert_allclose(numericDeriv, analyticDeriv, atol = 1e-6, rtol = 1e-4)
@codeDeps(assert_allclose)
def check_logJacDerivInput(outputTransform, input, x, eps):
    direction = randn(*np.shape(input))
    numericDeriv = (outputTransform.logJac(input + direction * eps, x) - outputTransform.logJac(input, x)) / eps
    analyticDeriv = np.dot(direction, outputTransform.logJacDerivInput(input, x))
    assert_allclose(numericDeriv, analyticDeriv, atol = 1e-6, rtol = 1e-4)

@codeDeps(assert_allclose, check_deriv, check_derivInput, check_derivParams,
    check_inv, check_logJac, check_logJacDeriv, check_logJacDerivInput,
    check_logJacDerivParams, xf.eval_local
)
def checkOutputTransform(outputTransform, shapeInput, shapeOutput, hasParams, eps, its, checkAdditional = None):
    outputTransformEvaled = xf.eval_local(repr(outputTransform))
    assert repr(outputTransform) == repr(outputTransformEvaled)
    if hasParams:
        params = outputTransform.params
        outputTransformParsed = outputTransform.parseAll(params)
        assert_allclose(outputTransformParsed.params, params)
        assert_allclose(outputTransformEvaled.params, params)
    for it in range(its):
        input = randn(*shapeInput)
        x = randn(*shapeOutput)
        transform = outputTransform.atInput(input)
        if checkAdditional is not None:
            checkAdditional(outputTransform, input, x, eps)
        if True:
            assert_allclose(outputTransform(input, x), transform(x))
        if True:
            assert_allclose(outputTransformEvaled(input, x), outputTransform(input, x))
        if hasParams:
            assert_allclose(outputTransformParsed(input, x), outputTransform(input, x))
        if True:
            check_deriv(transform, x, eps)
        if True:
            check_derivInput(outputTransform, input, x, eps)
        if hasParams:
            check_derivParams(transform, x, eps)
        if True:
            check_logJac(transform, x)
        if True:
            check_logJacDeriv(transform, x, eps)
        if True:
            check_logJacDerivInput(outputTransform, input, x, eps)
        if hasParams:
            check_logJacDerivParams(transform, x, eps)
        if True:
            check_inv(transform, x, randn(*shapeOutput))

@codeDeps(assert_allclose, checkOutputTransform, gen_ShiftOutputTransform,
    gen_SimpleOutputTransform, randBool, shapeRand,
    xf.DerivativeNotPositiveError, xf.SimpleOutputTransform, xf.TanhTransform1D
)
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

@codeDeps(TestOutputTransform, TestTransform)
def suite():
    return unittest.TestSuite([
        unittest.TestLoader().loadTestsFromTestCase(TestTransform),
        unittest.TestLoader().loadTestsFromTestCase(TestOutputTransform),
    ])

if __name__ == '__main__':
    unittest.main()
