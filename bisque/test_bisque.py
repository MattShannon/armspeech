"""Runs all tests for bisque."""

# Copyright 2011, 2012, 2013, 2014 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.


from __future__ import division

from bisque import test_queuer
from bisque import test_persist
from codedep import codeDeps

import unittest
import sys
import argparse

@codeDeps()
class ShouldNotPickle(object):
    pass

@codeDeps(ShouldNotPickle, test_persist.suite, test_queuer.suite)
def suite():
    return unittest.TestSuite([
        test_queuer.suite(),
        test_persist.suite(createShouldNotPickle = ShouldNotPickle),
    ])

@codeDeps(suite)
def main(rawArgs):
    parser = argparse.ArgumentParser(
        description = 'Runs all tests for bisque.',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--verbosity', dest = 'verbosity', type = int, default = 1,
        metavar = 'VERB',
        help = 'verbosity level'
    )
    args = parser.parse_args(rawArgs[1:])

    testResult = unittest.TextTestRunner(
        verbosity = args.verbosity
    ).run(suite())

    return 0 if testResult.wasSuccessful() else 1

if __name__ == '__main__':
    retCode = main(sys.argv)
    sys.exit(retCode)
