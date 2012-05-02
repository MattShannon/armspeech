"""Accumulators for estimating transforms."""

# Copyright 2011, 2012 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.


from __future__ import division

import dist as d
import transform as xf

import numpy as np

class TransformAccG(d.AccG):
    pass

class DerivInputTransformAccG(TransformAccG):
    def __init__(self, inputTransform, tag = None):
        self.inputTransform = inputTransform
        self.tag = tag

        self.derivParams = np.zeros([len(inputTransform.params)])

    def children(self):
        return []

    def add(self, (dist, input), output, occ = 1.0):
        inputT = self.inputTransform(input)
        self.derivParams += np.dot(
            self.inputTransform.derivParams(input),
            dist.logProbDerivInput(inputT, output)
        ) * occ

    def addAccSingle(self, acc):
        self.derivParams += acc.derivParams

    def logLikeSingle(self):
        # for gradient-based accumulation, the relevant logLike contribution is already counted during sub-dist accumulation
        # (FIXME : this may be a bit of a hack)
        return 0.0

    def derivParamsSingle(self):
        return self.derivParams


class OutputTransformAccG(d.AccG):
    pass

class DerivOutputTransformAccG(OutputTransformAccG):
    def __init__(self, outputTransform, tag = None):
        self.outputTransform = outputTransform
        self.tag = tag

        self.derivParams = np.zeros([len(outputTransform.params)])

    def children(self):
        return []

    def add(self, (dist, input), output, occ = 1.0):
        outputT = self.outputTransform(input, output)
        self.derivParams += np.dot(
            self.outputTransform.derivParams(input, output),
            dist.logProbDerivOutput(input, outputT)
        ) * occ
        self.derivParams += self.outputTransform.logJacDerivParams(input, output) * occ

    def addAccSingle(self, acc):
        self.derivParams += acc.derivParams

    def logLikeSingle(self):
        # for gradient-based accumulation, the relevant logLike contribution is already counted during sub-dist accumulation
        # (FIXME : this may be a bit of a hack)
        return 0.0

    def derivParamsSingle(self):
        return self.derivParams
