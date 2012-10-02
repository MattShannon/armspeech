"""Useful functions for object persistence."""

# Copyright 2011, 2012 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.


from __future__ import division

from codedep import codeDeps

import os
import cPickle as pickle
import hashlib
import tempfile

@codeDeps()
def loadPickle(location):
    with open(location, 'rb') as f:
        obj = pickle.load(f)
    return obj

@codeDeps(loadPickle)
def savePickle(location, obj):
    head, tail = os.path.split(location)
    with tempfile.NamedTemporaryFile(prefix = '.'+tail+'.', dir = head, mode='wb', delete = False) as f:
        tempLocation = f.name
        pickle.dump(obj, f, protocol = 2)

    # (FIXME : could remove below checks for speed, especially once we're dealing with large pickled files)
    # make sure we can read in again
    objAgain = loadPickle(tempLocation)
    # make sure pickled stream doesn't contain __main__ references
    #   (these can't be unpickled from a different script)
    with open(tempLocation, 'rb') as f:
        pickledString = f.read()
    if pickledString.find('__main__') != -1:
        raise pickle.PicklingError('objects to be pickled should be defined in a module and not in the currently-running script')

    # move into place atomically (at least on linux)
    # (FIXME : does windows behavior in fact still give correct result in our
    #   use cases?)
    os.rename(tempLocation, location)

@codeDeps()
def roundTrip(obj):
    return pickle.loads(pickle.dumps(obj, protocol = 2))

@codeDeps(roundTrip)
def checkRoundTrip(obj):
    assert pickle.dumps(roundTrip(obj), protocol = 2) == pickle.dumps(obj, protocol = 2)

@codeDeps()
def secHashString(strr):
    """Computes git-style hash of a string."""
    m = hashlib.sha1()
    m.update('blob '+str(len(strr))+'\0')
    m.update(strr)
    return m.hexdigest()

@codeDeps(roundTrip)
def secHashObject(obj):
    """Computes git-style hash of the pickled version of an object."""
    m = hashlib.sha1()
    # N.B. round-tripping here seems to help ensure that secHash is immune to
    #   round-tripping (that is, computing the secHash of an object gives the
    #   same result as pickling, unpickling, then computing the secHash).
    pickledObj = pickle.dumps(roundTrip(obj), protocol = 2)
    m.update('blob '+str(len(pickledObj))+'\0')
    m.update(pickledObj)
    return m.hexdigest()

@codeDeps()
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
