"""Representation of transforms.

Transforms are essentially functions with learnable parameters."""

# Copyright 2011, 2012, 2013 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.


from __future__ import division

from armspeech.util.mathhelp import logDet, reprArray
from minimize import solveByMinimize
from armspeech.util.lazy import lazyproperty
from codedep import codeDeps, ForwardRef

import math
import numpy as np
import armspeech.numpy_settings
import armspeech.util.mylinalg as mla

# (FIXME : current parsing potentially involves _a lot_ of list copying.
#   Refactor? (Probably never limiting factor in time or memory though.))

# FIXME : some of the asserts below should really be exceptions?

def eval_local(reprString):
    from numpy import array, zeros, dtype, float64

    return eval(reprString)

@codeDeps()
def parseConcat(parsers, params):
    outs = []
    paramsLeft = params
    for parser in parsers:
        out, paramsLeft = parser(paramsLeft)
        outs.append(out)
    return outs, paramsLeft

@codeDeps()
class DerivativeNotPositiveError(Exception):
    pass

@codeDeps()
class Transform(object):
    """Function of one argument with learnable parameters.

    Used to transform input to conditional distributions.
    """

    # (FIXME : should probably extend tree structure to transforms)
    def children(self):
        return []
    def mapChildren(self, mapChild):
        return self
    def createAccG(self, createAccChild):
        # (FIXME : ugly import!)
        import transform_acc as xfa
        return xfa.DerivInputTransformAccG(inputTransform = self, tag = self.tag)
    def paramsSingle(self):
        return self.params
    def paramsChildren(self, paramsChild):
        return []
    def parseSingle(self, params):
        return self.parse(params)
    def parseChildren(self, params, parseChild):
        return self, params
    def parseAll(self, params):
        transform, paramsLeft = self.parse(params)
        if len(paramsLeft) != 0:
            raise RuntimeError('extra parameters left after parsing complete')
        return transform
    def withTag(self, tag):
        """Set tag and return self.

        This is intended to be used immediately after object creation, such as:

            transform = SomeTransform([2.0, 3.0, 4.0]).withTag('hi')

        This is particularly important here since Transforms should be immutable.
        """
        self.tag = tag
        return self

@codeDeps(Transform)
class ConstantTransform(Transform):
    def __init__(self, value, tag = None):
        self.value = value
        self.tag = tag
    def __repr__(self):
        return 'ConstantTransform('+repr(self.value)+', tag = '+repr(self.tag)+')'
    @property
    def params(self):
        return np.array([])
    def parse(self, params):
        return ConstantTransform(self.value, tag = self.tag), params
    def __call__(self, x):
        return self.value
    def deriv(self, x):
        return np.zeros(np.shape(x) + np.shape(self.value))
    def derivParams(self, x):
        return np.zeros(np.shape(self.params) + np.shape(self.value))

@codeDeps(Transform)
class IdentityTransform(Transform):
    def __init__(self, tag = None):
        self.tag = tag
    def __repr__(self):
        return 'IdentityTransform(tag = '+repr(self.tag)+')'
    @property
    def params(self):
        return np.array([])
    def parse(self, params):
        return IdentityTransform(tag = self.tag), params
    def __call__(self, x):
        return x
    def deriv(self, x):
        if np.shape(x) == ():
            return 1.0
        else:
            assert x.ndim == 1
            return np.eye(len(x))
    def derivDeriv(self, x):
        assert np.shape(x) == ()
        return 0.0
    def derivParams(self, x):
        if np.shape(x) == ():
            return np.array([])
        else:
            return np.eye(0, len(x))
    def derivParamsDeriv(self, x):
        assert np.shape(x) == ()
        return np.array([])
    def logJac(self, x):
        return 0.0
    def logJacDeriv(self, x):
        return np.zeros(np.shape(x))
    def logJacDerivParams(self, x):
        return np.array([])
    def inv(self, y):
        return y

@codeDeps(Transform)
class DotProductTransform(Transform):
    def __init__(self, params, tag = None):
        self.params = params
        self.tag = tag
    def __repr__(self):
        return 'DotProductTransform('+repr(self.params)+', tag = '+repr(self.tag)+')'
    def parse(self, params):
        n = len(self.params)
        return DotProductTransform(params[:n], tag = self.tag), params[n:]
    def __call__(self, x):
        assert x.ndim == 1
        return np.dot(self.params, x)
    def deriv(self, x):
        assert x.ndim == 1
        return self.params
    def derivParams(self, x):
        assert x.ndim == 1
        return x

@codeDeps(Transform, lazyproperty, logDet, mla.inv, reprArray)
class LinearTransform(Transform):
    def __init__(self, mat, tag = None):
        assert mat.ndim == 2
        self.mat = mat
        self.tag = tag
    def __repr__(self):
        return 'LinearTransform(%s, tag=%r)' % (reprArray(self.mat), self.tag)
    @property
    def params(self):
        return np.reshape(self.mat, (-1,))
    def parse(self, params):
        n = len(self.params)
        matNew = np.reshape(params[:n], np.shape(self.mat))
        return LinearTransform(matNew, tag = self.tag), params[n:]
    def __call__(self, x):
        assert x.ndim == 1
        return np.dot(x, self.mat)
    def deriv(self, x):
        assert x.ndim == 1
        return self.mat
    def derivParams(self, x):
        dimIn, dimOut = np.shape(self.mat)
        assert np.shape(x) == (dimIn,)
        if dimIn == 0 or dimOut == 0:
            return np.eye(0, dimOut)
        else:
            return np.concatenate([ np.eye(dimOut) * xp for xp in x ], axis = 0)
    @lazyproperty
    def logDetMat(self):
        return logDet(self.mat)
    @lazyproperty
    def invMat(self):
        return mla.inv(self.mat)
    def logJac(self, x):
        assert x.ndim == 1
        return self.logDetMat
    def logJacDeriv(self, x):
        assert x.ndim == 1
        return np.zeros(np.shape(x))
    def logJacDerivParams(self, x):
        assert x.ndim == 1
        return np.reshape(np.transpose(self.invMat), (-1,))
    def inv(self, y):
        assert y.ndim == 1
        # FIXME : store some factorization instead of inverse and use that instead to compute x for any given y?
        return np.dot(y, self.invMat)

@codeDeps(Transform)
class FrozenTransform(Transform):
    """prevents params of transform from being exposed, so they won't be re-estimated"""
    def __init__(self, transform, tag = None):
        self.transform = transform
        self.tag = tag
    def __repr__(self):
        return 'FrozenTransform('+repr(self.transform)+', tag = '+repr(self.tag)+')'
    def __call__(self, x):
        return self.transform(x)
    def deriv(self, x):
        return self.transform.deriv(x)
    def logJac(self, x):
        return self.transform.logJac(x)
    def logJacDeriv(self, x):
        return self.transform.logJacDeriv(x)
    def inv(self, y):
        return self.transform.inv(y)

@codeDeps(Transform, mla.inv)
class InvertedTransform(Transform):
    def __init__(self, transform, tag = None):
        self.transform = transform
        self.tag = tag
    def __repr__(self):
        return 'InvertedTransform('+repr(self.transform)+', tag = '+repr(self.tag)+')'
    @property
    def params(self):
        return self.transform.params
    def parse(self, params):
        xf, paramsLeft = self.transform.parse(params)
        return InvertedTransform(xf, tag = self.tag), paramsLeft
    def __call__(self, y):
        return self.transform.inv(y)
    def deriv(self, y):
        derivOrig = self.transform.deriv(self.transform.inv(y))
        if np.shape(derivOrig) == ():
            return 1.0 / derivOrig
        else:
            return mla.inv(derivOrig)
    def derivParams(self, y):
        # FIXME : replace with a right division or something? (but N.B. matrices not vectors, so possible?)
        return -np.dot(
            self.transform.derivParams(self.transform.inv(y)),
            self.deriv(y)
        )
    def logJac(self, y):
        return -self.transform.logJac(self.transform.inv(y))
    def logJacDeriv(self, y):
        return -np.dot(
            self.transform.logJacDeriv(self.transform.inv(y)),
            self.deriv(y)
        )
    def logJacDerivParams(self, y):
        return -np.dot(
            self.derivParams(y),
            self.transform.logJacDeriv(self.transform.inv(y))
        ) - self.transform.logJacDerivParams(self.transform.inv(y))
    def inv(self, x):
        return self.transform(x)

@codeDeps(Transform)
class TransposeTransform(Transform):
    def __init__(self, tag = None):
        self.tag = tag
    def __repr__(self):
        return 'TransposeTransform(tag = '+repr(self.tag)+')'
    def __call__(self, x):
        return zip(*x)
    def logJac(self, x):
        return 0.0
    def inv(self, y):
        return zip(*y)

@codeDeps(Transform)
class AddBias(Transform):
    def __init__(self, tag = None):
        self.tag = tag
    def __repr__(self):
        return 'AddBias(tag = '+repr(self.tag)+')'
    def __call__(self, x):
        x = np.asarray(x)
        assert x.ndim == 1
        return np.append(x, 1.0)
    def deriv(self, x):
        x = np.asarray(x)
        assert x.ndim == 1
        n = len(x)
        return np.eye(n, n + 1)

@codeDeps(Transform)
class AddBiasVec(Transform):
    def __init__(self, tag = None):
        self.tag = tag
    def __repr__(self):
        return 'AddBiasVec(tag = '+repr(self.tag)+')'
    def __call__(self, x):
        inputLength, order = np.shape(x)
        return np.concatenate((x, np.ones((1, order))), axis = 0)
    def deriv(self, x):
        notyetimplemented

@codeDeps(Transform)
class MinusPrev(Transform):
    def __init__(self, tag = None):
        self.tag = tag
    def __repr__(self):
        return 'MinusPrev(tag = '+repr(self.tag)+')'
    def __call__(self, x):
        assert x.ndim == 1
        assert len(x) >= 1
        return -x[-1]
    def deriv(self, x):
        assert x.ndim == 1
        assert len(x) >= 1
        v = np.zeros(np.shape(x))
        v[-1] = -1.0
        return v

@codeDeps(Transform)
class Msd01ToVector(Transform):
    def __init__(self, tag = None):
        self.tag = tag
    def __repr__(self):
        return 'Msd01ToVector(tag = '+repr(self.tag)+')'
    def __call__(self, input):
        out = []
        for comp, x in input:
            if comp == 0:
                out.extend([1.0, 0.0])
            else:
                out.extend([0.0, x])
        return np.array(out)

@codeDeps(Transform)
class VectorizeTransform(Transform):
    def __init__(self, transform1D, tag = None):
        self.transform1D = transform1D
        self.tag = tag
    def __repr__(self):
        return 'VectorizeTransform('+repr(self.transform1D)+', tag = '+repr(self.tag)+')'
    @property
    def params(self):
        return self.transform1D.params
    def parse(self, params):
        xf, paramsLeft = self.transform1D.parse(params)
        return VectorizeTransform(xf, tag = self.tag), paramsLeft
    def __call__(self, x):
        assert x.ndim == 1
        return np.array(map(self.transform1D, x))
    def deriv(self, x):
        assert x.ndim == 1
        return np.diag(map(self.transform1D.deriv, x))
    def derivParams(self, x):
        assert x.ndim == 1
        if len(x) == 0:
            return np.eye(len(self.params), 0)
        else:
            return np.transpose(map(self.transform1D.derivParams, x))
    def logJac(self, x):
        assert x.ndim == 1
        return sum(map(self.transform1D.logJac, x))
    def logJacDeriv(self, x):
        assert x.ndim == 1
        return map(self.transform1D.logJacDeriv, x)
    def logJacDerivParams(self, x):
        assert x.ndim == 1
        if len(x) == 0:
            return np.zeros(np.shape(self.params))
        else:
            return np.sum(map(self.transform1D.logJacDerivParams, x), axis = 0)
    def inv(self, y):
        assert y.ndim == 1
        return np.array(map(self.transform1D.inv, y))

@codeDeps(Transform, solveByMinimize)
class Transform1D(Transform):
    def logJac(self, x):
        return math.log(abs(self.deriv(x)))
    def logJacDeriv(self, x):
        return self.derivDeriv(x) / self.deriv(x)
    def logJacDerivParams(self, x):
        return self.derivParamsDeriv(x) / self.deriv(x)
    def inv(self, y, length = -100):
        # (FIXME : could consider different starting points (e.g. 0.0))
        def F(x):
            return self(x), np.array([self.deriv(x)])
        return solveByMinimize(F, y, y, length = length)

@codeDeps(Transform1D)
class PolynomialTransform1D(Transform1D):
    def __init__(self, params, tag = None):
        self.params = params
        self.tag = tag
    def __repr__(self):
        return 'PolynomialTransform1D('+repr(self.params)+', tag = '+repr(self.tag)+')'
    def parse(self, params):
        n = len(self.params)
        return PolynomialTransform1D(params[:n], tag = self.tag), params[n:]
    def __call__(self, x):
        return sum([
            coeff * x ** power
            for power, coeff in enumerate(self.params)
        ])
    def deriv(self, x):
        return sum([
            coeff * (power + 1) * x ** power
            for power, coeff in enumerate(self.params[1:])
        ])
    def derivDeriv(self, x):
        return sum([
            coeff * (power + 1) * (power + 2) * x ** power
            for power, coeff in enumerate(self.params[2:])
        ])
    def derivParams(self, x):
        return np.array([
            x ** power
            for power, coeff in enumerate(self.params)
        ])
    def derivParamsDeriv(self, x):
        return np.array([
            power * x ** (power - 1) if power != 0 else 0.0
            for power, coeff in enumerate(self.params)
        ])

@codeDeps(Transform1D)
class ScaledSinhTransform1D(Transform1D):
    """scaled sinh transform (scaling parameterized slightly oddly, since mainly for testing)"""
    def __init__(self, a, tag = None):
        self.a = a
        self.tag = tag
    def __repr__(self):
        return 'ScaledSinhTransform1D('+repr(self.a)+', tag = '+repr(self.tag)+')'
    @property
    def params(self):
        return [self.a]
    def parse(self, params):
        aNew = params[0]
        return ScaledSinhTransform1D(aNew, tag = self.tag), params[1:]
    def __call__(self, x):
        return self.a * math.sinh(self.a * x)
    def deriv(self, x):
        return self.a * self.a * math.cosh(self.a * x)
    def derivDeriv(self, x):
        return self.a * self.a * self.a * math.sinh(self.a * x)
    def derivParams(self, x):
        return np.array([math.sinh(self.a * x) + self.a * x * math.cosh(self.a * x)])
    def derivParamsDeriv(self, x):
        return np.array([2.0 * self.a * math.cosh(self.a * x) + self.a * self.a * x * math.sinh(self.a * x)])
    def inv(self, y):
        return math.asinh(y / self.a) / self.a

@codeDeps(ForwardRef(lambda: TanhTransform1D), Transform1D)
class TanhTransformLogParam1D(Transform1D):
    def __init__(self, params, tag = None):
        self.params = params
        self.tag = tag

        p, q, c = params
        a = math.exp(p)
        b = math.exp(q)
        self.tt = TanhTransform1D([a, b, c])
        self.derivParamsConvert = np.array([a, b, 1.0])
    def __repr__(self):
        return 'TanhTransformLogParam1D('+repr(self.params)+', tag = '+repr(self.tag)+')'
    def parse(self, params):
        n = len(self.params)
        return TanhTransformLogParam1D(params[:n], tag = self.tag), params[n:]
    def __call__(self, x):
        return self.tt(x)
    def deriv(self, x):
        return self.tt.deriv(x)
    def derivDeriv(self, x):
        return self.tt.derivDeriv(x)
    def derivParams(self, x):
        return self.tt.derivParams(x) * self.derivParamsConvert
    def derivParamsDeriv(self, x):
        return self.tt.derivParamsDeriv(x) * self.derivParamsConvert
    def logJacDerivParams(self, x):
        """(override for mild efficiency improvement)"""
        return self.tt.logJacDerivParams(x) * self.derivParamsConvert
    def inv(self, y):
        return self.tt.inv(y)

@codeDeps(Transform1D)
class TanhTransform1D(Transform1D):
    def __init__(self, params, warn = False, tag = None):
        self.params = params
        self.warn = warn
        self.tag = tag

        self.a, self.b, self.c = params
        if warn and self.a <= 0.0:
            print 'NOTE: a =', self.a, '<= 0.0'
        if warn and self.b <= 0.0:
            print 'NOTE: b =', self.b, '<= 0.0'
    def __repr__(self):
        return 'TanhTransform1D('+repr(self.params)+', warn = '+repr(self.warn)+', tag = '+repr(self.tag)+')'
    def parse(self, params):
        n = len(self.params)
        return TanhTransform1D(params[:n], warn = self.warn, tag = self.tag), params[n:]
    def __call__(self, x):
        th = math.tanh(self.b * (x - self.c))
        return self.a * th
    def deriv(self, x):
        ch = math.cosh(self.b * (x - self.c))
        sh2 = 1.0 / ch / ch
        return self.a * self.b * sh2
    def derivDeriv(self, x):
        ch = math.cosh(self.b * (x - self.c))
        th = math.tanh(self.b * (x - self.c))
        sh2 = 1.0 / ch / ch
        return -2.0 * self.a * self.b * self.b * sh2 * th
    def derivParams(self, x):
        ch = math.cosh(self.b * (x - self.c))
        th = math.tanh(self.b * (x - self.c))
        sh2 = 1.0 / ch / ch
        return np.array([
            th,
            self.a * (x - self.c) * sh2,
            -self.a * self.b * sh2
        ])
    def derivParamsDeriv(self, x):
        ch = math.cosh(self.b * (x - self.c))
        th = math.tanh(self.b * (x - self.c))
        sh2 = 1.0 / ch / ch
        return np.array([
            self.b * sh2,
            self.a * sh2 * (1.0 - 2.0 * self.b * (x - self.c) * th),
            2.0 * self.a * self.b * self.b * sh2 * th
        ])
    def logJacDerivParams(self, x):
        """(override for mild efficiency improvement)"""
        th = math.tanh(self.b * (x - self.c))
        return np.array([
            1.0 / self.a,
            1.0 / self.b - (x - self.c) * th * 2.0,
            self.b * th * 2.0
        ])
    def inv(self, y):
        return math.atanh(y / self.a) / self.b + self.c

@codeDeps(Transform1D, parseConcat)
class SumTransform1D(Transform1D):
    def __init__(self, transforms, tag = None):
        self.transforms = transforms
        self.tag = tag
    def __repr__(self):
        return 'SumTransform1D('+repr(self.transforms)+', tag = '+repr(self.tag)+')'
    @property
    def params(self):
        return np.concatenate([ transform.params for transform in self.transforms ])
    def parse(self, params):
        xfs, paramsLeft = parseConcat([ transform.parse for transform in self.transforms ], params)
        return SumTransform1D(xfs, tag = self.tag), paramsLeft
    def __call__(self, x):
        return sum([ transform(x) for transform in self.transforms ])
    def deriv(self, x):
        return sum([ transform.deriv(x) for transform in self.transforms ])
    def derivDeriv(self, x):
        return sum([ transform.derivDeriv(x) for transform in self.transforms ])
    def derivParams(self, x):
        return np.concatenate([ transform.derivParams(x) for transform in self.transforms ])
    def derivParamsDeriv(self, x):
        return np.concatenate([ transform.derivParamsDeriv(x) for transform in self.transforms ])

@codeDeps(ForwardRef(lambda: TransformAtInput))
class OutputTransform(object):
    """Input-dependent invertible transform of output.

    An output transform takes an input and an output and returns a transformed
    output.
    For any given input the function from output to transformed output is
    invertible.
    """

    def atInput(self, input, tag = None):
        return TransformAtInput(self, input, tag = tag)
    # (FIXME : should probably extend tree structure to transforms)
    def children(self):
        return []
    def mapChildren(self, mapChild):
        return self
    def createAccG(self, createAccChild):
        # (FIXME : ugly import!)
        import transform_acc as xfa
        return xfa.DerivOutputTransformAccG(outputTransform = self, tag = self.tag)
    def paramsSingle(self):
        return self.params
    def paramsChildren(self, paramsChild):
        return []
    def parseSingle(self, params):
        return self.parse(params)
    def parseChildren(self, params, parseChild):
        return self, params
    def parseAll(self, params):
        transform, paramsLeft = self.parse(params)
        if len(paramsLeft) != 0:
            raise RuntimeError('extra parameters left after parsing complete')
        return transform
    def withTag(self, tag):
        """Set tag and return self.

        This is intended to be used immediately after object creation, such as:

            outputTransform = SomeOutputTransform([2.0, 3.0, 4.0]).withTag('hi')

        This is particularly important here since OutputTransforms should be immutable.
        """
        self.tag = tag
        return self

@codeDeps(Transform)
class TransformAtInput(Transform):
    def __init__(self, outputTransform, input, tag = None):
        self.outputTransform = outputTransform
        self.input = input
        self.tag = tag
    def __repr__(self):
        return 'TransformAtInput('+repr(self.outputTransform)+', '+repr(self.input)+', tag = '+repr(self.tag)+')'
    @property
    def params(self):
        return self.outputTransform.params
    def parse(self, params):
        xf, paramsLeft = self.outputTransform.parse(params)
        return TransformAtInput(xf, self.input, tag = self.tag), paramsLeft
    def __call__(self, x):
        return self.outputTransform(self.input, x)
    def deriv(self, x):
        return self.outputTransform.deriv(self.input, x)
    def derivParams(self, x):
        return self.outputTransform.derivParams(self.input, x)
    def logJac(self, x):
        return self.outputTransform.logJac(self.input, x)
    def logJacDeriv(self, x):
        return self.outputTransform.logJacDeriv(self.input, x)
    def logJacDerivParams(self, x):
        return self.outputTransform.logJacDerivParams(self.input, x)
    def inv(self, y):
        return self.outputTransform.inv(self.input, y)

@codeDeps(DerivativeNotPositiveError, OutputTransform)
class SimpleOutputTransform(OutputTransform):
    """Output transform that is input-independent.

    For 1D transforms, may wish to set checkDerivPositive1D to check derivative
    is positive everywhere it is evaluated (this gives no guarantees about its
    behaviour elsewhere).
    """
    def __init__(self, transform, checkDerivPositive1D = False, tag = None):
        self.transform = transform
        self.checkDerivPositive1D = checkDerivPositive1D
        self.tag = tag
    def __repr__(self):
        return 'SimpleOutputTransform('+repr(self.transform)+', checkDerivPositive1D = '+repr(self.checkDerivPositive1D)+', tag = '+repr(self.tag)+')'
    @property
    def params(self):
        return self.transform.params
    def parse(self, params):
        xf, paramsLeft = self.transform.parse(params)
        return SimpleOutputTransform(xf, checkDerivPositive1D = self.checkDerivPositive1D, tag = self.tag), paramsLeft
    def __call__(self, input, realOutput):
        return self.transform(realOutput)
    def deriv(self, input, realOutput):
        ret = self.transform.deriv(realOutput)
        if self.checkDerivPositive1D and ret <= 0.0:
            raise DerivativeNotPositiveError('derivative '+repr(ret)+' should be > 0.0')
        return ret
    def derivInput(self, input, realOutput):
        assert len(np.shape(input)) <= 1
        assert len(np.shape(realOutput)) <= 1
        return np.zeros(np.shape(input) + np.shape(realOutput))
    def derivParams(self, input, realOutput):
        return self.transform.derivParams(realOutput)
    def logJac(self, input, realOutput):
        return self.transform.logJac(realOutput)
    def logJacDeriv(self, input, realOutput):
        return self.transform.logJacDeriv(realOutput)
    def logJacDerivInput(self, input, realOutput):
        return np.zeros(np.shape(input))
    def logJacDerivParams(self, input, realOutput):
        return self.transform.logJacDerivParams(realOutput)
    def inv(self, input, modelledOutput):
        return self.transform.inv(modelledOutput)

@codeDeps(OutputTransform)
class ShiftOutputTransform(OutputTransform):
    def __init__(self, shift, tag = None):
        self.shift = shift
        self.tag = tag
    def __repr__(self):
        return 'ShiftOutputTransform('+repr(self.shift)+', tag = '+repr(self.tag)+')'
    @property
    def params(self):
        return self.shift.params
    def parse(self, params):
        xf, paramsLeft = self.shift.parse(params)
        return ShiftOutputTransform(xf, tag = self.tag), paramsLeft
    def __call__(self, input, realOutput):
        return realOutput + self.shift(input)
    def deriv(self, input, realOutput):
        if np.shape(realOutput) == ():
            return 1.0
        else:
            assert realOutput.ndim == 1
            return np.eye(len(realOutput))
    def derivInput(self, input, realOutput):
        return self.shift.deriv(input)
    def derivParams(self, input, realOutput):
        return self.shift.derivParams(input)
    def logJac(self, input, realOutput):
        return 0.0
    def logJacDeriv(self, input, realOutput):
        return np.zeros(np.shape(realOutput))
    def logJacDerivInput(self, input, realOutput):
        return np.zeros(np.shape(input))
    def logJacDerivParams(self, input, realOutput):
        return np.zeros(np.shape(self.params))
    def inv(self, input, modelledOutput):
        return modelledOutput - self.shift(input)
