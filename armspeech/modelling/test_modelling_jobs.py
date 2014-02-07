"""Unit tests for distributed job helper functions in modelling."""

# Copyright 2011, 2012, 2013, 2014 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.


from __future__ import division

from codedep import codeDeps

import unittest

@codeDeps()
class TestCorpus(unittest.TestCase):
    pass

@codeDeps()
class TestTrain(unittest.TestCase):
    pass

@codeDeps(TestCorpus, TestTrain)
def suite():
    return unittest.TestSuite([
        unittest.TestLoader().loadTestsFromTestCase(TestCorpus),
        unittest.TestLoader().loadTestsFromTestCase(TestTrain),
    ])

if __name__ == '__main__':
    unittest.main()
