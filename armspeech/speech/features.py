"""Representation, I/O and utility functions for acoustic features."""

# Copyright 2011, 2012, 2013, 2014, 2015 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.

import os
import logging
import struct
import math
import numpy as np
import numpy.linalg as la
import itertools
import subprocess

from codedep import codeDeps

import armspeech.numpy_settings

@codeDeps()
def getMcdConstant():
    return 10.0 / math.log(10.0) * math.sqrt(2.0)

_mcdConstant = getMcdConstant()

@codeDeps(getMcdConstant)
def stdCepDist(synthVec, actualVec):
    return _mcdConstant * la.norm(np.asarray(synthVec)[1:] - np.asarray(actualVec)[1:])
@codeDeps(getMcdConstant)
def stdCepDistIncZero(synthVec, actualVec):
    return _mcdConstant * la.norm(np.asarray(synthVec) - np.asarray(actualVec))

@codeDeps()
class NoneEncoder(object):
    def __init__(self):
        self.decode = None
        self.encode = None

@codeDeps(NoneEncoder)
class Stream(object):
    def __init__(self, name, order, encoder = NoneEncoder()):
        self.name = name
        self.order = order
        self.encoder = encoder

    def __repr__(self):
        return 'Stream('+repr(self.name)+', '+repr(self.order)+', '+repr(self.encoder)+')'

# (FIXME : move this to htk_io (generalizing slightly)?)
@codeDeps()
class AcousticSeqIo(object):
    def __init__(self, dir, vecSeqIos, exts, encoders):
        self.dir = dir
        self.vecSeqIos = vecSeqIos
        self.exts = exts
        self.encoders = encoders

        self.numStreams = len(self.vecSeqIos)

        assert len(self.vecSeqIos) == self.numStreams
        assert len(self.exts) == self.numStreams
        assert len(self.encoders) == self.numStreams

    def writeFiles(self, uttId, acousticSeq):
        elemSeqs = zip(*acousticSeq)
        assert len(elemSeqs) == self.numStreams

        for streamIndex in range(self.numStreams):
            ext = self.exts[streamIndex]
            vecSeqFile = os.path.join(self.dir, '%s.%s' % (uttId, ext))

            encode = self.encoders[streamIndex].encode
            elemSeq = elemSeqs[streamIndex]
            vecSeq = map(encode, elemSeq)
            self.vecSeqIos[streamIndex].writeFile(vecSeqFile, vecSeq)

    def readFiles(self, uttId):
        elemSeqs = []
        for streamIndex in range(self.numStreams):
            ext = self.exts[streamIndex]
            vecSeqFile = os.path.join(self.dir, '%s.%s' % (uttId, ext))

            vecSeq = self.vecSeqIos[streamIndex].readFile(vecSeqFile)
            decode = self.encoders[streamIndex].decode
            elemSeq = map(decode, vecSeq)
            elemSeqs.append(elemSeq)

        acousticSeq = zip(*elemSeqs)
        return acousticSeq

@codeDeps()
class Msd01Encoder(object):
    def __init__(self, specialValue):
        self.specialValue = specialValue

    def decode(self, xs):
        x, = xs
        if x == self.specialValue:
            return 0, None
        else:
            return 1, x
    def encode(self, value):
        comp, x = value
        if comp == 0:
            return [self.specialValue]
        else:
            return [x]

@codeDeps()
def doHtsDemoWaveformGeneration(scriptsDir, synthOutDir, basenames, logFile = None):
    """HTS-demo-with-STRAIGHT-style waveform generation.

    N.B. assumes files to synthesize are <basename>.{mgc,lf0,bap} in synthOutDir.
    Also assumes a matching Config.pm configuration file.
    """
    args = ['/usr/bin/perl', os.path.join(scriptsDir, 'gen_wave.pl'), os.path.join(scriptsDir, 'Config.pm'), synthOutDir] + basenames
    p = subprocess.Popen(args, stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
    logOutput = p.communicate()[0]
    if p.returncode != 0:
        logging.warning('waveform generation failed (exit code '+str(p.returncode)+')')
    if logFile is not None:
        with open(logFile, 'w') as f:
            f.write(logOutput)
