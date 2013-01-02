"""Unit tests for persistence helper functions."""

# Copyright 2011, 2012, 2013 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.


from __future__ import division

import persist
from filehelp import TempDir
from codedep import codeDeps

import unittest
import cPickle as pickle
import os

@codeDeps()
class ShouldNotPickle(object):
    pass

@codeDeps(ShouldNotPickle, TempDir, persist.loadPickle, persist.savePickle,
    persist.secHashFile, persist.secHashObject, persist.secHashString
)
class TestPersist(unittest.TestCase):
    def setUp(self):
        self.createShouldNotPickle = ShouldNotPickle

    def test_pickling(self):
        l = [1, 2, 3]
        d = dict()
        d['a'] = l
        d['b'] = l
        tempDir = TempDir()
        def loc(suffix):
            return os.path.join(tempDir.location, suffix)
        location1 = loc('d1.pickle')
        location2 = loc('d2.pickle')
        persist.savePickle(location1, d)
        persist.savePickle(location2, d)
        dAgain1 = persist.loadPickle(location1)
        dAgain2 = persist.loadPickle(location2)
        assert d == dAgain1 == dAgain2
        tempDir.remove()
    def test_shouldNotPickle(self):
        tempDir = TempDir()
        def loc(suffix):
            return os.path.join(tempDir.location, suffix)
        obj = self.createShouldNotPickle()
        self.assertRaises(pickle.PicklingError, persist.savePickle, loc('obj.pickle'), obj)
        tempDir.remove()
    def test_secHash_consistent(self):
        """Tests that secHashFile of a pickled object is the same as secHashObject of the object."""
        l = [1, 2, 3]
        d = dict()
        d['a'] = l
        d['b'] = l
        tempDir = TempDir()
        def loc(suffix):
            return os.path.join(tempDir.location, suffix)
        persist.savePickle(loc('d.pickle'), d)
        assert persist.secHashFile(loc('d.pickle')) == persist.secHashObject(d)
        tempDir.remove()
    def test_secHashString_characterization(self):
        """Simple characterization test for secHashString so we will know if it ever changes."""
        assert persist.secHashString('') == 'e69de29bb2d1d6434b8b29ae775ad8c2e48c5391'
        assert persist.secHashString('"') == '9d68933c44f13985b9eb19159da6eb3ff0e574bf'
        assert persist.secHashString('abc') == 'f2ba8f84ab5c1bce84a7b441cb1959cfc7093b7f'
        assert persist.secHashString('aRvo;ui') == 'af7aa2bc77f03b961a67b633f93674331eab36c8'
    def test_secHashObject_characterization(self):
        """Simple characterization test for secHashObject so we will know if it ever changes."""
        l = [1, 2, 3]
        d = dict()
        d['a'] = l
        d['b'] = l
        assert persist.secHashObject(0) == '27b9b44c6234ee94be9ee6f1af3aca8f77c42194'
        assert persist.secHashObject(1) == 'aaabe6ddf3884ad6332f7ba21ebde77cfa56020f'
        assert persist.secHashObject(2) == 'f90d0d7dda12125e6e903c7fe294ef5a18b5c38d'
        assert persist.secHashObject([]) == 'a766f8f1a671877187d2e4cc90987785f7323bde'
        assert persist.secHashObject(None) == '6ae75066bc250a2c28d6871cbb2e5e6089800440'
        assert persist.secHashObject(dict()) == '10c1a6c4ca5103c9310a788b6ac71b1eb03026f3'
        assert persist.secHashObject(l) == '4af396d95ed2d87f7330f8418f1d7619623e5f42'
        assert persist.secHashObject(d) == 'bebad045b5f2d023efed1d4fc1840a28c532789e'

@codeDeps(TestPersist)
def suite(createShouldNotPickle = None):
    if createShouldNotPickle is None:
        return unittest.TestLoader().loadTestsFromTestCase(TestPersist)
    else:
        class CustomTestPersist(TestPersist):
            def setUp(self):
                self.createShouldNotPickle = createShouldNotPickle

        return unittest.TestLoader().loadTestsFromTestCase(CustomTestPersist)

if __name__ == '__main__':
    unittest.main()
