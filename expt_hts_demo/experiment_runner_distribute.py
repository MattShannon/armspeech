#!/usr/bin/python -u

"""Command-line tool to run example experiments on a grid or similar."""

# Copyright 2011, 2012, 2013 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.


from __future__ import division

from expt_hts_demo import experiment
from bisque import distribute
import bisque.queuer as qr
from bisque import sge_queuer

import os
import sys
import time

def main(rawArgs):
    buildRepo = qr.BuildRepo(base = os.path.join('expt_hts_demo', 'repo'))

    outDir = os.path.join('expt_hts_demo', 'out.d')
    synthOutDir = os.path.join(outDir, 'synth')
    figOutDir = os.path.join(outDir, 'fig')
    if not os.path.exists(synthOutDir):
        raise RuntimeError('dir %s should already exist' % synthOutDir)
    if not os.path.exists(figOutDir):
        raise RuntimeError('dir %s should already exist' % figOutDir)
    synthOutDirArt = distribute.FixedDirArtifact(synthOutDir)
    figOutDirArt = distribute.FixedDirArtifact(figOutDir)

    finalArtifacts = experiment.doMonophoneSystemJobSet(synthOutDirArt,
                                                        figOutDirArt)

    useLocalQueuer = False
    if useLocalQueuer:
        queuer = qr.LocalQueuer(buildRepo = buildRepo)
        queuer.generateArtifacts(finalArtifacts, verbosity = 1)
    else:
        with sge_queuer.MockSgeQueuer(buildRepo = buildRepo) as queuer:
            queuer.generateArtifacts(finalArtifacts, verbosity = 1)

    print 'final values:'
    for art in finalArtifacts:
        print '\tart (%s) -> %s' % (art.secHash(), art.loadValue(buildRepo))

if __name__ == '__main__':
    main(sys.argv)
