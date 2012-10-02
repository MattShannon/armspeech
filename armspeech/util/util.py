"""General purpose utility functions."""

# Copyright 2011, 2012 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.


from __future__ import division

from codedep import codeDeps

@codeDeps()
def orderedDictRepr(orderedKeys, dictIn):
    assert len(orderedKeys) == len(dictIn)
    return '{'+', '.join([ repr(key)+': '+repr(dictIn[key]) for key in orderedKeys ])+'}'
