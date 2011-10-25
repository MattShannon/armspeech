"""Runs all tests."""

# Copyright 2011 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.


from __future__ import division

import test_dist
import test_distribute
import test_mathhelp
import test_minimize
import test_persist
import test_transform

import unittest
import sys
import argparse

class ShouldNotPickle(object):
    pass

def suite(deepTest = False):
    return unittest.TestSuite([
        test_dist.suite(deepTest = deepTest),
        test_distribute.suite(),
        test_mathhelp.suite(),
        test_minimize.suite(),
        test_persist.suite(createShouldNotPickle = ShouldNotPickle),
        test_transform.suite(),
    ])

def main(rawArgs):
    parser = argparse.ArgumentParser(description = 'Runs all tests for armspeech.')
    parser.add_argument(
        '--verbosity', dest = 'verbosity', type = int, default = 1, metavar = 'VERB',
        help = 'verbosity level (default: 1)'
    )
    parser.add_argument(
        '--deep', dest = 'deepTest', action = 'store_true',
        help = 'include very slow tests'
    )
    args = parser.parse_args(rawArgs[1:])

    testResult = unittest.TextTestRunner(verbosity = args.verbosity).run(suite(deepTest = args.deepTest))

    return 0 if testResult.wasSuccessful() else 1

if __name__ == '__main__':
    retCode = main(sys.argv)
    sys.exit(retCode)
