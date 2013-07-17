"""Implements code-level dependency tracking."""

# Copyright 2011, 2012, 2013 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.


from __future__ import division

from codedep.decorators import codeHash, codeDeps, ForwardRef, codedepEvalThunk
from codedep.compute import getHash
