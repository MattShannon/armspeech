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
def readParamFile(paramFile, paramOrder, decode = None):
    floatLittleEndian = struct.Struct('<'+''.join([ 'f' for i in range(paramOrder) ]))
    with open(paramFile, 'rb') as f:
        while True:
            bytes = f.read(paramOrder * 4)
            if bytes == '':
                break
            curr = np.array(floatLittleEndian.unpack(bytes))
            yield curr if decode is None else decode(curr)
# (FIXME : use instead of above??)
# (N.B. perhaps surprisingly, it seems quite a bit slower (43 sec vs 60 sec in one test)
#   than readParamFile!)
@codeDeps()
def readParamFileAlt(paramFile, paramOrder):
    return np.reshape(
        np.fromfile(paramFile, dtype = np.float32),
        (-1, paramOrder)
    ).astype(np.float64)

@codeDeps()
def readParamFileDoubleAlt(paramFile, paramOrder):
    return np.reshape(
        np.fromfile(paramFile, dtype = np.float64),
        (-1, paramOrder)
    )

@codeDeps()
def writeParamFile(outSeq, paramFile, paramOrder, encode = None):
    floatLittleEndian = struct.Struct('<'+''.join([ 'f' for i in range(paramOrder) ]))
    with open(paramFile, 'wb') as f:
        for out in outSeq:
            curr = out if encode is None else encode(out)
            bytes = floatLittleEndian.pack(*curr)
            f.write(bytes)

@codeDeps()
def writeParamFileAlt(outSeq, paramFile):
    outSeq.astype(np.float32).tofile(paramFile)

@codeDeps()
def writeParamFileDoubleAlt(outSeq, paramFile):
    outSeq.astype(np.float64).tofile(paramFile)

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

@codeDeps(readParamFile)
def readAcousticGen(streams, paramFileFor):
    return itertools.izip(*[
        readParamFile(paramFileFor(stream), stream.order, stream.encoder.decode)
        for stream in streams
    ])

@codeDeps(writeParamFile)
def writeAcousticSeq(outSeq, streams, paramFileFor):
    for stream, outSeqStream in zip(streams, zip(*outSeq)):
        writeParamFile(outSeqStream, paramFileFor(stream), stream.order, stream.encoder.encode)

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
