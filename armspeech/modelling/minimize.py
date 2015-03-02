* 1.0 """Function minimization using conjugate gradients."""

# Copyright 2011, 2012, 2013, 2014, 2015 Matt Shannon
# Copyright 2005-2010 Carl Edward Rasmussen and Hannes Nickisch

# This file is part of armspeech.
# See `License` for details of license and warranty.

from __future__ import division

import sys
import logging
import traceback
from numpy import array, zeros, shape, dot, isnan, isinf, isreal, sqrt, finfo, double
from numpy.random import randn

from codedep import codeDeps, ForwardRef

import armspeech.numpy_settings

@codeDeps(ForwardRef(lambda: checkGradAt))
def checkGrad(f, dim, drawInput = lambda dim: 2.0 * randn(dim), numPoints = 500, relTol = 1e-2, absTol = 1e-3):
    """checks analytic gradient returned by f agrees with numerical gradient at randomly chosen points (fairly weak test but still very useful)"""
    for i in range(numPoints):
        X = drawInput(dim)
        checkGradAt(f, X, relTol, absTol)

@codeDeps()
def checkGradAt(f, X, relTol, absTol):
    """checks analytic gradient returned by f agrees with numerical gradient at the specified point"""
    if len(shape(X)) != 1:
        raise RuntimeError('argument of function to minimize must be a vector')
    dim = len(X)
    f0, df0 = f(X)
    e = 1e-8
    dfn = zeros(dim)
    for i in range(dim):
        Xp = X.copy()
        Xp[i] += e
        fp, dfp = f(Xp)
        dfn[i] = (fp - f0) / e
    if any(abs((df0 - dfn) / dfn) > relTol) and any(abs(df0 - dfn) > absTol):
        maxRelDiff = max(abs((df0 - dfn) / dfn))
        maxAbsDiff = max(abs(df0 - dfn))
        print
        print 'NOTE: maxRelDiff =', maxRelDiff, '( relTol =', relTol, ')'
        print 'NOTE: maxAbsDiff =', maxAbsDiff, '( absTol =', absTol, ')'
        raise RuntimeError('analytic and numeric derivatives differ ('+repr(df0)+' vs '+repr(dfn)+')')

@codeDeps()
def minimize(f, X, length, red = 1.0, verbosity = 0):
    if len(shape(X)) != 1:
        raise RuntimeError('argument of function to minimize must be a vector')

    INT = 0.1
    EXT = 3.0
    MAX = 20
    RATIO = 10.0
    SIG = 0.1; RHO = SIG/2.0

    if length>0:
        S='Linesearch'
    else:
        S='Function evaluation'

    i = 0
    ls_failed = 0
    f0, df0 = f(X)
    if verbosity >= 2:
        print 'minimize:', '%s %6i;\tValue %4.6e' % (S, i, f0)
    fX = [f0]
    i = i + (length<0)
    s = -df0; d0 = -dot(s, s)
    x3 = red/(1.0-d0)

    while i < abs(length):
        i = i + (length>0)

        X0 = X; F0 = f0; dF0 = df0
        if length>0:
            M = MAX
        else:
            M = min(MAX, -length-i)

        while True:
            x2 = 0.0; f2 = f0; d2 = d0; f3 = f0; df3 = df0
            success = False
            while not success and M > 0:
                # from Carl Rasmussen's minimize.m code:
                # "During extrapolation, the 'f' function may fail either with an error
                # or returning Nan or Inf, and minimize should handle this gracefully."
                M = M - 1; i = i + (length<0)
                try:
                    f3, df3 = f(X+x3*s)
                except KeyboardInterrupt:
                    raise
                except Exception, e:
                    logging.warning(e.__class__.__name__+' during function evaluation during minimize')
                    if verbosity >= 3:
                        print 'minimize:', '-'*60
                        print 'minimize: exception during function evaluation:'
                        traceback.print_exc(file = sys.stdout)
                        print 'minimize:', '-'*60
                    x3 = (x2+x3)/2.0
                else:
                    if isnan(f3) or isinf(f3) or any(isnan(df3)) or any(isinf(df3)):
                        print 'NOTE: nan or inf during function evaluation during minimize'
                        x3 = (x2+x3)/2.0
                    else:
                        success = True
            if f3 < F0:
                X0 = X+x3*s; F0 = f3; dF0 = df3
            d3 = dot(df3, s)
            if d3 > SIG*d0 or f3 > f0+x3*RHO*d0 or M == 0:
                break
            x1 = x2; f1 = f2; d1 = d2
            x2 = x3; f2 = f3; d2 = d3
            A = 6.0*(f1-f2)+3.0*(d2+d1)*(x2-x1)
            B = 3.0*(f2-f1)-(2.0*d1+d2)*(x2-x1)
            x3 = x1-d1*(x2-x1)**2/(B+sqrt(B*B-A*d1*(x2-x1)))
            if not isreal(x3) or isnan(x3) or isinf(x3) or x3 < 0.0:
                x3 = x2*EXT
            elif x3 > x2*EXT:
                x3 = x2*EXT
            elif x3 < x2+INT*(x2-x1):
                x3 = x2+INT*(x2-x1)

        while (abs(d3) > -SIG*d0 or f3 > f0+x3*RHO*d0) and M > 0:
            if d3 > 0.0 or f3 > f0+x3*RHO*d0:
                x4 = x3; f4 = f3; d4 = d3
            else:
                x2 = x3; f2 = f3; d2 = d3
            if f4 > f0:
                x3 = x2-(0.5*d2*(x4-x2)**2)/(f4-f2-d2*(x4-x2))
            else:
                A = 6.0*(f2-f4)/(x4-x2)+3.0*(d4+d2)
                B = 3.0*(f4-f2)-(2.0*d2+d4)*(x4-x2)
                x3 = x2+(sqrt(B*B-A*d2*(x4-x2)**2)-B)/A
            if isnan(x3) or isinf(x3):
                x3 = (x2+x4)/2.0
            x3 = max(min(x3, x4-INT*(x4-x2)),x2+INT*(x4-x2))
            f3, df3 = f(X+x3*s)
            if f3 < F0:
                X0 = X+x3*s; F0 = f3; dF0 = df3
            M = M - 1; i = i + (length<0)
            d3 = dot(df3, s)

        if abs(d3) < -SIG*d0 and f3 < f0+x3*RHO*d0:
            X = X+x3*s; f0 = f3; fX.append(f0)
            if verbosity >= 2:
                print 'minimize:', '%s %6i;\tValue %4.6e' % (S, i, f0)
            s = (dot(df3, df3) - dot(df0, df3))/dot(df0, df0)*s - df3
            df0 = df3
            d3 = d0; d0 = dot(df0, s)
            if d0 > 0.0:
                s = -df0; d0 = -dot(s, s)
            x3 = x3 * min(RATIO, d3/(d0-finfo(double).tiny))
            ls_failed = 0
        else:
            X = X0; f0 = F0; df0 = dF0
            if ls_failed or i > abs(length):
                break
            s = -df0; d0 = -dot(s, s)
            x3 = 1.0/(1.0-d0)
            ls_failed = 1
    return X, fX, i

@codeDeps()
def solveToMinimize(F, a, convertFrom1D = False):
    """returns a function f whose global minima are precisely where F(x) = a"""
    def f(x):
        if convertFrom1D:
            x = x[0]
        F0, dF0 = F(x)
        y = F0 - a
        return 0.5 * dot(y, y), dot(y, dF0)
    return f

@codeDeps()
class NoSolutionError(Exception):
    pass

@codeDeps()
class DidNotConvergeError(Exception):
    pass

@codeDeps(DidNotConvergeError, NoSolutionError, minimize, solveToMinimize)
def solveByMinimize(F, a, x0, length, red = 1.0, verbosity = 0, solvedThresh = 1e-8):
    """solves F(x) = a iteratively, starting at x0"""
    convertFrom1D = (shape(x0) == ())
    if convertFrom1D:
        x0 = array([x0])
    f = solveToMinimize(F, a, convertFrom1D = convertFrom1D)
    sol, fs, lengthUsed = minimize(f, x0, length, red, verbosity = verbosity)
    solValue = f(sol)[0]
    if lengthUsed == length:
        raise DidNotConvergeError('solver did not converge (length = '+repr(length)+')')
    elif solValue > solvedThresh:
        raise NoSolutionError('no solution found (probable local minimum of '+repr(solValue)+' not 0.0)')
    if convertFrom1D:
        sol = sol[0]
    return sol
