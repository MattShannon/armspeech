#!/usr/bin/python -u

"""Convenience script to print a pickled object."""

# Copyright 2011, 2012, 2013 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.


from __future__ import division

import sys
import cPickle as pickle

def main(args):
    assert len(args) == 2
    with open(args[1], 'rb') as f:
        obj = pickle.load(f)
    print obj

if __name__ == '__main__':
    main(sys.argv)
