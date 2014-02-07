"""Unit tests for sequence helper functions."""

# Copyright 2011, 2012, 2013, 2014 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.


from __future__ import division

import iterhelp
from codedep import codeDeps

import unittest
import random

@codeDeps()
def gen_list(length):
    return [ random.choice('ABCDEFGHIJKLM') for i in range(length) ]

@codeDeps(gen_list, iterhelp.chunkList)
class TestIterHelp(unittest.TestCase):
    def test_chunkList(self, its = 100):
        for it in range(its):
            length = random.randint(0, 20)
            numChunks = random.randint(1, 20)
            xs = gen_list(length)
            chunks = iterhelp.chunkList(xs, numChunks)
            assert len(chunks) == numChunks
            # chunks should partition the original list
            assert [ x for chunk in chunks for x in chunk ] == xs
            # chunks should all be roughly the same size
            sizes = set([ len(chunk) for chunk in chunks ])
            assert len(sizes) <= 2

@codeDeps(TestIterHelp)
def suite():
    return unittest.TestLoader().loadTestsFromTestCase(TestIterHelp)

if __name__ == '__main__':
    unittest.main()
