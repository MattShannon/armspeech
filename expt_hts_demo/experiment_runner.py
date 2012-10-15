#!/usr/bin/python -u

"""Command-line runner for example experiments."""

# Copyright 2011, 2012 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.


from __future__ import division

from expt_hts_demo import experiment

import sys
import tempfile
import traceback
import numpy as np
import armspeech.numpy_settings

import matplotlib
matplotlib.use('Agg')

# (FIXME : could change this to 'raise' if we extensively test all experiments
#   to make sure they never fail incorrectly with this setting)
np.seterr(all = 'ignore')
np.set_printoptions(precision = 17, linewidth = 10000)

def main(rawArgs):
    outDir = tempfile.mkdtemp(dir = 'expt_hts_demo', prefix = 'out.')

    try:
        experiment.run(outDir)
    except:
        traceback.print_exc()
        print
        print '(to delete dir:)'
        print 'rm -r', outDir
        sys.exit(1)

if __name__ == '__main__':
    main(sys.argv)
