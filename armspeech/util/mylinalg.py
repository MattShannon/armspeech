"""Slightly customized versions of numpy / scipy linalg methods.

The standard numpy and scipy linalg routines both cope badly with
0-dimensional matrices or vectors. This module wraps several standard
routines to check for these special cases.
"""

# Copyright 2011, 2012, 2013 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.


from __future__ import division

from codedep import codeDeps

import numpy as np
import armspeech.numpy_settings
import numpy.linalg as la
import scipy.linalg as sla

@codeDeps()
def inv(*args, **kwargs):
    a = args[0] if len(args) > 0 else kwargs['a']
    if np.shape(a) == (0, 0):
        return np.eye(0)
    else:
        return la.inv(*args, **kwargs)

@codeDeps()
def pinv(*args, **kwargs):
    a = args[0] if len(args) > 0 else kwargs['a']
    if np.shape(a) == (0, 0):
        return np.eye(0)
    else:
        return la.pinv(*args, **kwargs)

@codeDeps()
def solve(*args, **kwargs):
    a = args[0] if len(args) > 0 else kwargs['a']
    b = args[1] if len(args) > 1 else kwargs['b']
    if np.shape(a) == (0, 0) and np.shape(b)[0] == 0:
        return np.zeros(np.shape(b))
    else:
        return sla.solve(*args, **kwargs)

@codeDeps()
def cholesky(*args, **kwargs):
    a = args[0] if len(args) > 0 else kwargs['a']
    if np.shape(a) == (0, 0):
        return np.eye(0)
    else:
        return sla.cholesky(*args, **kwargs)

@codeDeps()
def cho_solve(*args, **kwargs):
    c, lower = args[0]
    b = args[1] if len(args) > 1 else kwargs['b']
    if np.shape(c) == (0, 0) and np.shape(b) == (0,):
        return np.zeros(np.shape(b))
    else:
        return sla.cho_solve(*args, **kwargs)

# (not strictly speaking in linalg but whatever)
@codeDeps()
def tensordot(*args, **kwargs):
    a = args[0] if len(args) > 0 else kwargs['a']
    b = args[1] if len(args) > 1 else kwargs['b']
    axes = args[2] if len(args) > 2 else kwargs['axes']
    # (FIXME : specific to axes being an integer. Make more general.) (N.B. default numpy routine copes fine with axes == 0)
    if np.shape(axes) == () and axes > 0 and sum(np.shape(a)[-axes:]) == 0 and sum(np.shape(b)[:axes]) == 0:
        return np.zeros(np.shape(a)[:-axes] + np.shape(b)[axes:])
    else:
        return np.tensordot(*args, **kwargs)
