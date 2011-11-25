"""Useful functions for object persistence."""

# Copyright 2011 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.


from __future__ import division
from __future__ import with_statement

import os
import cPickle as pickle
import hashlib

def loadPickle(location):
    with open(location, 'rb') as f:
        obj = pickle.load(f)
    return obj

def savePickle(location, obj):
    with open(location, 'wb') as f:
        pickle.dump(obj, f, protocol = 2)
    # (FIXME : could remove below checks for speed, especially once we're dealing with large pickled files)
    # make sure we can read in again
    objAgain = loadPickle(location)
    # make sure pickled stream doesn't contain __main__ references
    #   (these can't be unpickled from a different script)
    with open(location, 'rb') as f:
        pickledString = f.read()
    if pickledString.find('__main__') != -1:
        raise pickle.PicklingError('objects to be pickled should be defined in a module and not in the currently-running script')

def roundTrip(obj):
    return pickle.loads(pickle.dumps(obj, protocol = 2))

def checkRoundTrip(obj):
    assert pickle.dumps(pickle.loads(pickle.dumps(obj, protocol = 2)), protocol = 2) == pickle.dumps(obj, protocol = 2)

def secHashObject(obj):
    """Computes git-style hash of the pickled version of an object."""
    m = hashlib.sha1()
    pickledObj = pickle.dumps(obj, protocol = 2)
    m.update('blob '+str(len(pickledObj))+'\0')
    m.update(pickledObj)
    return m.hexdigest()

def secHashFile(location, readChunkSize = 2 ** 20):
    """Computes git-style hash of a file."""
    m = hashlib.sha1()
    size = os.path.getsize(location)
    m.update('blob '+str(size)+'\0')
    with open(location, 'rb') as f:
        while True:
            chunk = f.read(readChunkSize)
            if not chunk:
                break
            m.update(chunk)
    return m.hexdigest()
