"""Corpus helper functions for distributed jobs."""

# Copyright 2011, 2012, 2013 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.


from __future__ import division

from armspeech.util.iterhelp import chunkList
from bisque.distribute import liftLocal, lit, lift
from codedep import codeDeps

@codeDeps(chunkList)
def getUttIdChunks(corpus, numChunks):
    return chunkList(corpus.trainUttIds, numChunks)

@codeDeps(getUttIdChunks, liftLocal, lit)
def getUttIdChunkArts(corpusArt, numChunksLit = lit(1)):
    numChunks = numChunksLit.litValue
    return liftLocal(getUttIdChunks, numOut = numChunks)(corpusArt,
                                                         numChunksLit)
