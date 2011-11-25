"""Runs all tests."""

# Copyright 2011 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.


from __future__ import division

from armspeech.bisque import test_queuer
from armspeech.modelling import test_dist
from armspeech.modelling import test_minimize
from armspeech.modelling import test_transform
from armspeech.util import test_mathhelp
from armspeech.util import test_persist

import unittest
import sys
import argparse

class ShouldNotPickle(object):
    pass

def suite(deepTest = False):
    return unittest.TestSuite([
        test_queuer.suite(),
        test_dist.suite(deepTest = deepTest),
        test_minimize.suite(),
        test_transform.suite(),
        test_mathhelp.suite(),
        test_persist.suite(createShouldNotPickle = ShouldNotPickle),
    ])

def main(rawArgs):
    parser = argparse.ArgumentParser(
        description = 'Runs all tests for armspeech.',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--verbosity', dest = 'verbosity', type = int, default = 1, metavar = 'VERB',
        help = 'verbosity level'
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
