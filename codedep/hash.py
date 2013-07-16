"""Specifies hash functions."""

# Copyright 2011, 2012, 2013 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.


from __future__ import division

import hashlib

def hashString(strr):
    """Computes git-style hash of a string."""
    m = hashlib.sha1()
    m.update('blob '+str(len(strr))+'\0')
    m.update(strr)
    return m.hexdigest()
