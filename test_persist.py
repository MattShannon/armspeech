"""Unit tests for persistence framework."""

# Copyright 2011 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.


from __future__ import division

import persist

import unittest
import cPickle as pickle
import os
import sys
import tempfile
import shutil

class TempSimpleRepo(persist.SimpleRepo):
    def __init__(self):
        self.base = tempfile.mkdtemp()
    def remove(self):
        shutil.rmtree(self.base)
    def __del__(self):
        if os.path.isdir(self.base):
            sys.stderr.write('\nWARNING: temporary directory '+self.base+' not deleted.  You probably want to do this manually after looking at its contents.\n')

class ShouldNotPickle(object):
    pass

class TestPersist(unittest.TestCase):
    def test_pickling(self):
        d = dict()
        d['a'] = 1
        d['b'] = 2
        repo = TempSimpleRepo()
        location1 = repo.newLocation()
        location2 = repo.newLocation()
        persist.savePickle(location1, d)
        persist.savePickle(location2, d)
        dAgain1 = persist.loadPickle(location1)
        dAgain2 = persist.loadPickle(location2)
        assert d == dAgain1 == dAgain2
        repo.remove()
    def test_shouldNotPickle(self):
        repo = TempSimpleRepo()
        obj = ShouldNotPickle()
        self.assertRaises(pickle.PicklingError, persist.savePickle, repo.newLocation(), obj)
        repo.remove()

if __name__ == '__main__':
    unittest.main()
