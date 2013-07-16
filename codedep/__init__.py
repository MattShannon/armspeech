"""Implements code-level dependency tracking."""

# Copyright 2011, 2012, 2013 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.


from __future__ import division

from decorators import codeHash, codeDeps, ForwardRef, codedepEvalThunk
from compute import getHash
