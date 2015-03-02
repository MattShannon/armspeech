"""Runs all tests for armspeech."""

# Copyright 2011, 2012, 2013, 2014, 2015 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.

import unittest
import sys
import argparse

from codedep import codeDeps

from armspeech.modelling import test_dist
from armspeech.modelling import test_minimize
from armspeech.modelling import test_transform
from armspeech.modelling import test_wnet
from armspeech.modelling import test_modelling_jobs
from armspeech.util import test_iterhelp
from armspeech.util import test_mathhelp
from armspeech.util import test_memoize

@codeDeps(test_dist.suite, test_iterhelp.suite, test_mathhelp.suite,
    test_memoize.suite, test_minimize.suite, test_modelling_jobs.suite,
    test_transform.suite, test_wnet.suite
)
def suite(deepTest = False):
    return unittest.TestSuite([
        test_dist.suite(deepTest = deepTest),
        test_minimize.suite(),
        test_transform.suite(),
        test_wnet.suite(),
        test_modelling_jobs.suite(),
        test_iterhelp.suite(),
        test_mathhelp.suite(),
        test_memoize.suite(),
    ])

@codeDeps(suite)
def main(rawArgs):
    parser = argparse.ArgumentParser(
        description = 'Runs all tests for armspeech.',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--verbosity', dest = 'verbosity', type = int, default = 1,
        metavar = 'VERB',
        help = 'verbosity level'
    )
    parser.add_argument(
        '--deep', dest = 'deepTest', action = 'store_true',
        help = 'include very slow tests'
    )
    args = parser.parse_args(rawArgs[1:])

    testResult = unittest.TextTestRunner(
        verbosity = args.verbosity
    ).run(suite(deepTest = args.deepTest))

    return 0 if testResult.wasSuccessful() else 1

if __name__ == '__main__':
    retCode = main(sys.argv)
    sys.exit(retCode)
