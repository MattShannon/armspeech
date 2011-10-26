"""Simple object persistence framework built on top of pickle."""

# Copyright 2011 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.


from __future__ import division
from __future__ import with_statement

import os
import cPickle as pickle
import random

# FIXME : replace with shelve?

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

class Repo(object):
    def newLocation(self):
        abstract

class SimpleRepo(Repo):
    def __init__(self, base, createDirIfNece = True):
        self.base = base
        if createDirIfNece and not os.path.isdir(base):
            os.mkdir(base)
        if not os.path.isdir(base):
            raise RuntimeError('directory does not exist: '+base)
    def newId(self):
        # FIXME : use even more entropy to make a collision almost impossible
        return ''.join([ random.choice('0123456789') for i in range(7) ])
    def newLocation(self):
        return os.path.join(self.base, self.newId())

class Artifact(object):
    def __init__(self, location):
        self.location = location
